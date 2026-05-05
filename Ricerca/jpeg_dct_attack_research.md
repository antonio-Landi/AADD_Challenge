# Analisi ricerca JPEG/EOT/DCT per AADD-2026

Data analisi: 2026-05-05.

## Fonti

Paper locali:

- `/Users/anto/AADD_Challenge/Ricerca/mlsec17_paper_54.md`: Richard Shin, Dawn Song, "JPEG-resistant Adversarial Images".
- `/Users/anto/AADD_Challenge/Ricerca/athalye18b.md`: Athalye et al., "Synthesizing Robust Adversarial Examples".
- `/Users/anto/AADD_Challenge/Ricerca/A_Novel_Adversarial_Gray-Box_Attack_on_DCT-Based_Face_Deepfake_Detectors.md`: Guarnera et al., "A Novel Adversarial Gray-Box Attack on DCT-Based Face Deepfake Detectors".

Paper citati nelle note e verificati online:

- Reich et al., "Differentiable JPEG: The Devil Is in the Details", WACV 2024: https://openaccess.thecvf.com/content/WACV2024/html/Reich_Differentiable_JPEG_The_Devil_Is_in_the_Details_WACV_2024_paper.html
- Hussain et al., "Adversarial Deepfakes: Evaluating Vulnerability of Deepfake Detectors to Adversarial Examples", WACV 2021: https://openaccess.thecvf.com/content/WACV2021/html/Hussain_Adversarial_Deepfakes_Evaluating_Vulnerability_of_Deepfake_Detectors_to_Adversarial_Examples_WACV_2021_paper.html

Contesto locale:

- `/Users/anto/AADD_Challenge/AADD_2026_RULEBOOK.md`
- `/Users/anto/AADD_Challenge/AADD_2026_evaluation.py`

## Conclusione rapida

La tua lettura e corretta: per AADD-2026 non dobbiamo trattare JPEG come un test finale da fare "dopo". JPEG deve entrare nella generazione dell'attacco.

La direzione piu solida e:

```text
attacco target Real
  + EOT su JPEG / resize / processing leggero
  + loss con ViT RGB e DenseNet-DCT
  + perturbazione concentrata su componenti luma low-mid frequency
  + selezione per immagine secondo score atteso post-processing
```

Pero c'e un punto critico: il paper DCT gray-box e molto rilevante concettualmente, ma non coincide con il nostro detector 2026. Guarnera et al. manipolano statistiche AC di DCT locali 8x8 modellate con distribuzioni Laplaciane. Il detector ufficiale locale `densenet121_dct`, invece, fa:

```text
RGB -> grayscale -> resize max 256 -> center crop 128 -> DCT 2D globale -> log(abs(DCT)+1e-6) -> DenseNet121
```

Quindi l'idea "rendere lo spettro fake piu simile a quello real" e utile, ma l'algoritmo specifico dei 63 beta 8x8 non va copiato direttamente. Per noi e meglio usare un vincolo/regularizer spettrale costruito sul preprocess ufficiale AADD, oppure ottimizzare direttamente attraverso una DCT differenziabile equivalente al modello 2026.

## Cosa JPEG fa davvero al nostro attacco

JPEG cambia il problema in tre modi:

1. Elimina o altera molte perturbazioni high-frequency.
   Gli attacchi pixel-level classici spesso funzionano perche mettono piccoli segnali distribuiti ad alta frequenza. JPEG quantizza proprio quelle zone in modo aggressivo.

2. Sposta il dominio rilevante verso luminanza e frequenza.
   JPEG lavora in YCbCr; il canale Y viene preservato piu dei canali cromatici. Il nostro DCT detector 2026 usa grayscale, quindi una perturbazione solo cromatica puo essere fragile o invisibile al DCT.

3. Rompe il gradiente.
   Rounding, floor e clipping danno gradienti nulli quasi ovunque. Senza DiffJPEG o STE, ottimizzare "attraverso JPEG" e praticamente impossibile.

Per AADD-2026, il punto non e aumentare epsilon. Il punto e mettere l'energia adversarial in frequenze che:

- sopravvivono alla quantizzazione JPEG;
- vengono ancora viste dal `vit_b_16`;
- cambiano in modo utile la mappa `log(abs(DCT))` del `densenet121_dct`;
- non peggiorano troppo SSIM/LPIPS.

In pratica: niente fiducia cieca nelle alte frequenze. Meglio low-mid frequency, soprattutto in luminanza, con EOT.

## Paper 1: Shin & Song 2017

### Idea

Shin e Song mostrano che JPEG come difesa non basta se l'attaccante e adattivo. Invece di ottimizzare:

```text
C(x_adv)
```

ottimizzano:

```text
C(JPEG_diff(x_adv, q))
```

dove `JPEG_diff` approssima JPEG con operazioni differenziabili.

### Dettagli utili

La pipeline differenziabile replica:

- RGB -> YCbCr;
- chroma subsampling via average pooling 2x2;
- blocchi 8x8;
- DCT lineare;
- quantizzazione;
- approssimazione del rounding;
- decoding inverso.

La parte critica e il rounding. Loro usano:

```text
round_approx(x) = round(x) + (x - round(x))^3
```

Questo mantiene un forward vicino al rounding, ma con gradiente non nullo.

La seconda idea forte e l'ensemble su piu quality factor:

```text
q in {25, 50, 75, no JPEG}
```

Ottimizzare su un singolo `q` produce attacchi iperspecializzati: possono funzionare a QF 25 e fallire senza JPEG, o viceversa. L'ensemble riduce questa fragilita.

### Risultato concettuale

Nel targeted I-FGSM, l'attacco adattivo su ensemble migliora drasticamente la sopravvivenza alla compressione. Il messaggio importante non e il numero specifico su ImageNet, ma il principio:

```text
se la valutazione include JPEG, JPEG deve stare nel grafo di ottimizzazione.
```

### Limiti per noi

- Il setting e ImageNet, non deepfake detection.
- L'approssimazione JPEG e storica; Reich 2024 mostra che manca diversi dettagli del JPEG reale.
- Non considera LPIPS.
- Non considera un detector DCT come quello AADD.

### Cosa prendiamo

Da implementare come baseline concettuale:

```text
loss = mean_q loss_classifier(JPEG_diff_q(x_adv), target=Real)
```

con `q` campionati o pesati, includendo sempre anche identita/no-JPEG.

## Paper 2: Athalye et al. 2018, EOT

### Idea

Expectation over Transformations dice: se vuoi un esempio adversarial robusto a una famiglia di trasformazioni, non devi ottimizzare l'immagine singola, ma l'aspettativa sulle trasformazioni.

Forma:

```text
max E_t [log P(target | t(x_adv))]
subject to E_t [d(t(x_adv), t(x))] < epsilon
```

In pratica, a ogni step si campionano trasformazioni `t` e si fa backprop attraverso quelle trasformazioni.

### Perche e perfetto per AADD-2026

Il rulebook 2026 parla di compressione e social-media-like processing. Quindi la vera immagine vista dai classifier finali potrebbe essere:

```text
t(x_adv)
```

non `x_adv` pulita.

Per noi la distribuzione `T` dovrebbe includere, in modo graduale:

- identita;
- JPEG QF 95, 90, 85, 80, 75;
- resize leggero e ritorno;
- crop/center-crop coerente con i modelli;
- blur leggerissimo;
- forse rumore leggero solo in fase avanzata.

### Limite importante

EOT allarga il problema. Se `T` e troppo ampia, il gradiente diventa medio, meno specializzato e puo richiedere perturbazioni piu visibili. Athalye et al. lo dicono chiaramente: piu grande e la distribuzione, piu grande tende a essere il budget necessario.

### Cosa prendiamo

Non partire con un EOT enorme. Strategia consigliata:

```text
fase 1: identity + JPEG {95, 90, 85}
fase 2: aggiungi JPEG {80, 75}
fase 3: aggiungi resize leggero
fase 4: aggiungi blur/noise solo se serve
```

Questo evita di indebolire subito l'attacco sui modelli pubblici.

## Paper 3: Guarnera et al., DCT gray-box attack

### Idea

Il paper attacca detector basati su feature DCT senza conoscere i parametri del classifier. L'attaccante conosce il feature extractor: statistiche degli AC coefficient DCT su blocchi 8x8.

Il modello statistico e:

```text
AC_i ~ Laplace(0, beta_i)
```

Ogni immagine e rappresentata da 63 parametri:

```text
beta_1, ..., beta_63
```

L'attacco cerca di trasformare una fake image affinche il suo vettore beta assomigli a quello di una real image o di un centroid real.

### Algoritmo base

Per ogni AC mode `i`:

```text
q_i = beta_i_real / beta_i_fake
AC_i_fake <- q_i * AC_i_fake
```

Poi applica IDCT per tornare in pixel space.

### Problema del metodo base

Applicare trasformazioni DCT blocco per blocco produce blocking artifacts molto visibili.

La soluzione proposta:

- generare 64 versioni shiftate dell'immagine, una per ogni offset 8x8;
- applicare l'attacco a ogni versione;
- riallineare le immagini;
- combinare con median filter lungo la dimensione delle versioni;
- ritagliare i bordi;
- applicare una rifinitura finale.

Questo riduce gli artefatti perche gli errori di blocco non sono allineati fra le 64 versioni.

### Reference profile

Invece di usare una sola immagine real, il paper costruisce `k` centroidi nel feature space delle real images con k-means. Poi sceglie il centroid piu vicino alla fake image.

Trade-off:

- pochi centroidi: spostamento piu forte verso real, attacco piu efficace, piu rischio visivo;
- molti centroidi: target piu vicino alla fake, SSIM piu alta, attacco meno forte.

### Risultati utili

Il paper riporta:

- attacco efficace su SVM, Random Forest, CNN e ViT quando questi usano feature DCT/statistiche frequenziali;
- LPIPS spesso sotto 0.05 quando SSIM si avvicina a 1;
- robustezza buona a JPEG in molti setting;
- debolezza chiara contro downscaling forte e noise forte.

Post-processing:

- HQ: robusto a JPEG fino a QF 80, lieve calo a QF 90; downscaling 50% e il caso peggiore.
- LQ: robustezza inferiore; JPEG QF alti e rumore possono degradare di piu; downscaling resta molto dannoso.
- External real profile puo migliorare efficacia/robustezza a costo di un piccolo calo percettivo.

### Punto critico per AADD-2026

Questo paper non attacca esattamente il nostro modello.

Il suo target:

```text
8x8 block DCT -> 63 beta Laplaciani -> classifier
```

Il nostro target locale:

```text
grayscale -> resize/crop -> DCT globale 128x128 -> log(abs(DCT)+1e-6) -> DenseNet
```

Quindi:

- non possiamo aspettarci che il rescaling dei 63 beta funzioni direttamente;
- possiamo pero usare l'idea come regularizer spettrale;
- possiamo costruire un "real spectral profile" coerente con il preprocess AADD, non con i blocchi 8x8.

### Cosa prendiamo

Tre idee da riusare:

1. Fare sembrare il fake piu real nel dominio frequenziale.
2. Usare centroidi real, non un unico target medio.
3. Gestire gli artefatti di blocco/struttura con combinazione o ottimizzazione soft.

Per noi, versione adattata:

```text
L_spectrum = distance(profile_AADD_DCT(x_adv), nearest_real_centroid)
```

dove `profile_AADD_DCT` puo essere:

- radial band energies della DCT globale;
- mean/std per bande low/mid/high;
- log-magnitude map downsampled;
- oppure feature intermedie del `densenet121_dct`.

## Paper 4: Reich et al. 2024

### Idea

Reich et al. dicono: non basta rendere JPEG "piu o meno differenziabile"; i dettagli del JPEG reale contano. Confrontano approssimazioni esistenti e propongono una DiffJPEG piu fedele.

Differenziabile rispetto a:

- immagine;
- JPEG quality;
- quantization tables;
- parametri di conversione colore.

### Dettagli importanti

Rispetto a Shin/Song, modellano meglio:

- scala reale delle quantization tables;
- floor della scala della quantization table;
- floor delle quantization tables;
- clipping differenziabile delle tabelle;
- clipping differenziabile dell'output;
- variante STE con forward piu fedele e backward surrogate.

Risultato riportato:

- migliore approssimazione del JPEG OpenCV su tutto il range di quality;
- +3.47 dB PSNR medio rispetto al miglior metodo precedente;
- fino a +9.51 dB in compressione forte;
- gradienti migliori negli attacchi FGSM/IFGSM.

### Cosa prendiamo

Se implementiamo JPEG-aware attack, meglio:

- usare una DiffJPEG moderna tipo Reich;
- oppure usare una STE validata con forward reale/simile a OpenCV/PIL e backward surrogate;
- validare sempre contro JPEG reale, non solo contro `JPEG_diff`.

Regola pratica:

```text
train con DiffJPEG
select con JPEG reale PIL/OpenCV
```

Se l'attacco funziona solo nella DiffJPEG approssimata ma fallisce con JPEG reale, e overfitting alla simulazione.

## Paper 5: Hussain et al. WACV 2021

### Idea

Questo paper porta EOT direttamente nel deepfake detection. Mostra che un attacco semplice puo battere detector CNN su video non compressi, ma perde molta efficacia dopo compressione. La variante robusta ottimizza sotto trasformazioni.

Trasformazioni usate:

- Gaussian blur;
- Gaussian noise;
- translation;
- downsize/upsize.

Per white-box robust, campionano 12 trasformazioni per iterazione: tre per ciascuna famiglia.

### Risultato utile

Nel frame-by-frame setting:

- white-box non robusto: quasi perfetto su raw, ma peggiora su MJPEG;
- white-box robusto: mantiene successo molto alto anche su MJPEG;
- a pari distorsione, robust white-box migliora il successo su compressione;
- anche in H.264 forte mantiene successi alti.

Nel sequence-model setting:

- robust white-box batte fortemente il non robust dopo compressione;
- robust black-box migliora ma resta piu difficile.

### Cosa prendiamo

La robustezza a compressione non e gratis:

- aumenta distorsione;
- richiede piu gradient samples;
- puo abbassare performance pulita se dosata male.

Ma e necessaria se la valutazione finale applica processing.

Per AADD-2026:

```text
EOT non serve solo per JPEG.
Serve anche come surrogate della variabilita dei preprocess finali.
```

## Sintesi: cosa significa per il nostro metodo

### 1. JPEG-aware e obbligatorio

La pipeline base dovrebbe ottimizzare:

```text
L_attack = E_t [ L_model(t(x_adv), Real) ]
```

dove `t` include identita e JPEG.

Non basta:

```text
genera attacco -> comprimi -> spera
```

### 2. DCT-aware non significa solo "aggiungere DCT loss"

Per il modello ufficiale 2026 dobbiamo riprodurre il suo preprocess:

```text
grayscale
resize a 256 se max > 256
center crop 128
DCT 2D globale
log(abs(.)+1e-6)
DenseNet
```

La loss DCT piu diretta e semplicemente la target loss del modello:

```text
L_dct = CE(densenet_dct(preprocess_dct(x_adv)), Real)
```

Poi possiamo aggiungere un regularizer spettrale ispirato a Guarnera:

```text
L_spectral_profile = distance(profile(x_adv), real_centroid)
```

Ma deve essere un regularizer, non per forza il motore principale.

### 3. Low-mid frequency e luminanza sono la zona piu promettente

La perturbazione deve passare due filtri:

- JPEG preserva soprattutto basse e medie frequenze, in particolare Y/luminanza;
- il DCT detector usa grayscale, quindi vede soprattutto cambi di luminanza.

Ma le basse frequenze sono visibili. Quindi la zona migliore probabilmente e:

```text
luma + mid-low / mid frequencies + regioni testurizzate del volto
```

Da evitare come unico segnale:

- high frequency pura;
- chroma-only;
- bordi estremi non visti dal center crop;
- pattern 8x8 troppo regolari, perche peggiorano LPIPS/SSIM e possono essere compressi/visibili.

### 4. Il detector DCT AADD ha una fragilita specifica

`log(abs(DCT)+1e-6)` rende molto sensibili i coefficienti piccoli. Ma JPEG puo azzerare o quantizzare proprio quei coefficienti. Quindi non dobbiamo costruire un attacco che dipende da micro-coefficienti instabili.

Meglio cercare direzioni che:

- restano sopra la soglia effettiva di quantizzazione;
- sopravvivono a resize/crop;
- cambiano bande aggregate, non singoli coefficienti fragili.

### 5. La selezione per immagine e fondamentale

La formula AADD moltiplica similarita per successo. Se un candidato fallisce su un classifier, quel pezzo vale zero.

Per ogni immagine conviene generare piu candidati:

- candidato pulito: attacco senza JPEG, alta qualita;
- candidato JPEG-EOT leggero;
- candidato JPEG-EOT forte;
- candidato DCT-heavy/frequency-aware;
- eventualmente candidato latente o spectral-profile.

Poi scegliere:

```text
argmax mean_t mean_model [ similarity(x, cand) * indicator(model(t(cand)) == Real) ]
```

con `similarity = 0.5 * SSIM + 0.5 * (1 - LPIPS)` come nello script locale.

## Proposta operativa concreta

### Fase A: baseline JPEG-EOT

Implementare un attacco iterativo target Real:

```text
L = w_vit * CE(vit(T(x_adv)), Real)
  + w_dct * CE(dct(T(x_adv)), Real)
  + lambda_l2 * ||x_adv - x||
  + lambda_lpips * LPIPS(x_adv, x)
```

Per iniziare:

```text
T = {identity, JPEG 95, JPEG 90, JPEG 85}
```

Poi:

```text
T = {identity, JPEG 95, 90, 85, 80, resize 0.9/1.0}
```

### Fase B: parametro della perturbazione in frequenza/luma

Invece di ottimizzare pixel RGB libero, provare:

```text
delta_rgb = y_to_rgb(IDCT(mask_freq * theta))
```

oppure piu semplice:

```text
delta <- low/mid-pass filtered(delta)
```

Questo forza la perturbazione a stare in componenti piu JPEG-stabili.

### Fase C: DCT-profile regularizer

Costruire un profilo real coerente col modello AADD:

```text
profile(x) = radial_bands(log(abs(DCT_128(gray_crop(x)))+1e-6))
```

Precomputare centroidi real da FFHQ/CelebA-HQ o un dataset real disponibile:

```text
centroids = kmeans(profile(real_images), k)
target = nearest_centroid(profile(fake))
```

Aggiungere:

```text
L_profile = ||profile(x_adv) - target||_2
```

Attenzione: questo puo aiutare il DCT, ma potrebbe danneggiare il ViT. Va pesato e testato.

### Fase D: validazione dura con JPEG reale

Per ogni candidato valutare con:

```text
identity
PIL/OpenCV JPEG q=95,90,85,80,75
resize 0.9 -> back
resize 0.75 -> back
JPEG + resize
```

La DiffJPEG serve per il gradiente, ma la decisione finale deve usare encoding reale.

### Fase E: score per immagine

Per ogni immagine:

```text
score_internal(cand) =
  mean_over_transforms(
    official_similarity(x, cand) *
    sum_models indicator(model(t(cand)) == Real)
  )
```

Scegliere il candidato con score interno massimo, non quello con perturbazione minima.

## Esperimenti prioritari

1. Clean under JPEG.
   Valutare immagini originali e immagini solo JPEG-compresse sui due modelli pubblici. Se JPEG da solo sposta il DCT verso Real per alcune immagini, e informazione utile.

2. Attack senza JPEG vs con JPEG.
   Generare MI-FGSM/PGD target Real senza EOT e misurare quanto crolla dopo JPEG.

3. Shin-style ensemble QF.
   Implementare `identity + DiffJPEG(q)` e confrontare q singolo vs ensemble.

4. Reich-style validation.
   Confrontare DiffJPEG training output contro JPEG reale PIL/OpenCV su un subset: ASR e similarity devono essere coerenti.

5. DCT-only vs ViT-only vs joint.
   Capire se i gradienti dei due modelli collaborano o si ostacolano.

6. Luma-only vs RGB-free.
   Testare se luma-only migliora DCT/JPEG robustness senza distruggere ViT.

7. Frequency mask sweep.
   Provare low, mid-low, mid, high. Misurare:
   - ASR clean;
   - ASR after JPEG;
   - SSIM;
   - LPIPS.

8. DCT-profile regularizer.
   Testare con e senza centroidi real. Se aiuta solo DCT ma rompe ViT, tenerlo come candidato separato.

9. Downscaling.
   Il DCT gray-box paper mostra che downscaling 50% e molto distruttivo. Per AADD, testare almeno 0.75 e 0.5, anche se non sappiamo quanto sara forte il processing ufficiale.

## Verdict

La strada piu promettente non e "JPEG oppure DCT", ma:

```text
JPEG-EOT come robustezza
+ DCT-aware loss come target
+ spectral/profile regularization come aiuto
+ candidate selection come assicurazione
```

Se la tua idea riguarda il manipolare direttamente frequenze/DCT, la domanda chiave sara: manipola il dominio giusto?

Per AADD-2026 il dominio giusto non e genericamente "DCT 8x8 tipo JPEG", ma:

```text
quello che resta dopo JPEG/resize
e che viene visto dalla DCT globale 128x128 del detector
```

Quindi io non partirei copiando Guarnera 1:1. Lo userei come ispirazione per un regularizer o per generare candidati DCT-heavy, mentre il motore principale dovrebbe restare un attacco differenziabile joint ViT+DCT con EOT JPEG.

