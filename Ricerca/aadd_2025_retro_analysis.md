# Analisi retrospettiva AADD-2025

Data analisi: 2026-05-05.

## Fonti usate

- `/Users/anto/AADD_Challenge/Ricerca/summary_aadd25.md`: paper overview ufficiale AADD-2025.
- `/Users/anto/AADD_Challenge/Ricerca/mr-cas.md`: paper MR-CAS, primo classificato.
- `/Users/anto/AADD_Challenge/Ricerca/mig-cow-paper.md`: paper Safe AI / MIG-COW, secondo classificato.
- Paper RoMa, terzo classificato: https://dl.acm.org/doi/10.1145/3746027.3761984; copia open access Fraunhofer: https://publica-rest.fraunhofer.de/server/api/core/bitstreams/9511f8cb-756d-4fc2-853e-ed21882f3f91/content.
- `/Users/anto/AADD_Challenge/AADD_2026_RULEBOOK.md`: confronto operativo con la challenge 2026.

## Sintesi esecutiva

AADD-2025 era una challenge di attacco adversarial contro detector di immagini deepfake. I partecipanti dovevano modificare immagini sintetiche in modo che fossero classificate come `real`, mantenendo alta similarita visiva. I modelli di valutazione erano quattro: due rilasciati ai partecipanti, ResNet50 e DenseNet121 RGB, e due blind, ViT-B/16 e DenseNet121-DCT.

La classifica non premia separatamente il miglior ASR o la miglior qualita: premia la somma della similarita solo sui casi in cui l'attacco riesce. Questo spiega bene la top 3:

| Rank | Team | Strategia centrale | SSIMS | ASS | FS |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | MR-CAS | DDIM inversion + ottimizzazione latente + MI-FGSM/ensemble | 0.742 | 0.672 | 2740 |
| 2 | Safe AI | MIG-COW: integrated gradients + consenso/ortogonale | 0.915 | 0.528 | 2709 |
| 3 | RoMa | rumore distribuito globale + surrogate ViT/EfficientNet | 0.934 | 0.509 | 2679 |

Il massimo teorico 2025 era circa `1403 immagini * 4 classifier = 5612`, assumendo SSIM 1 e successo su tutti. I primi tre sono attorno al 48-49% del massimo teorico. Questo e importante: anche i migliori non hanno "risolto" il black-box transfer, hanno trovato compromessi diversi fra successo e similarita.

La lezione principale e che il problema difficile non e battere i modelli pubblici. Quasi tutti i metodi forti riescono a farlo. Il vero collo di bottiglia e produrre perturbazioni che sopravvivano a cambi di architettura, cambi di dominio di input, e quest'anno anche compressione/post-processing.

## Contesto della challenge 2025

### Dati

Il dataset era composto da immagini deepfake divise in HQ e LQ, generate con modelli GAN e diffusion. Le immagini fake provenivano dal dataset WILD; le real usate per training/contesto derivavano da FFHQ e CelebA-HQ. Secondo i paper dei team, il test set comprendeva 1403 immagini fake: 693 HQ e 710 LQ.

Generatori citati:

- Diffusion/HQ: Adobe Firefly, DeepAI, Flux 1.1 Pro, Hotpot AI, Nvidia SanaPAG, Stable Diffusion 3.5, Tencent Hunyuan.
- Diffusion/LQ: DeepAI, Flux.1, Freepik, Hotpot AI, Nvidia SanaPAG, Stable Diffusion Attend-and-Excite, Tencent Hunyuan.
- GAN/HQ e LQ: StyleGAN2, StyleGAN3.

### Modelli

I modelli target erano:

- ResNet50 RGB, pubblico.
- DenseNet121 RGB, pubblico.
- ViT-B/16, blind.
- DenseNet121-DCT, blind.

Gia nel 2025 il DenseNet-DCT introduceva un cambio di dominio importante: non guarda direttamente l'immagine RGB, ma una rappresentazione frequenziale. Questo rende fragili molti attacchi ottimizzati solo nello spazio pixel/RGB.

### Scoring

La formula 2025 sommava, per ogni immagine e classifier:

```text
SSIM(original, adversarial) * indicator(predizione == real)
```

Quindi:

- se l'attacco fallisce, quella coppia immagine/classifier vale zero;
- se riesce, il contributo e proporzionale alla SSIM;
- un metodo con SSIM molto alta ma ASR basso puo perdere contro un metodo piu aggressivo;
- un metodo con ASR alta ma SSIM bassa puo ugualmente perdere.

Questa dinamica e visibile nei casi estremi: MILab ottiene SSIM 0.994 ma ASS 0.020, quindi score 110; VYAKRITI 2.0 ottiene ASS 0.615 ma SSIM 0.298, quindi score 1041. Il "centro di massa" della classifica sta nel prodotto fra successo e similarita, non in uno dei due valori isolati.

## Analisi dei primi tre metodi

### 1. MR-CAS: attacco latente via diffusion inversion

MR-CAS ha vinto con lo score piu alto, ma non con la miglior SSIM. La scelta e stata piu aggressiva: sacrificare una parte di similarita, ottenendo un ASS piu alto.

Elementi chiave:

- usa Stable Diffusion e DDIM inversion per mappare l'immagine nel latent space;
- ottimizza direttamente un latent a timestep basso, congelando la parte precedente del processo per preservare semantica e struttura;
- usa una descrizione generata da un vision-language model come guidance testuale;
- applica strategie classiche di trasferibilita dentro il framework latente: MI-FGSM, ensemble loss, data augmentation;
- aggiunge vincoli di consistenza, in particolare L1, per limitare distorsioni;
- usa perturbation radius crescente: parte piu conservativo e aumenta finche l'attacco riesce.

Impostazioni riportate:

- DDIM sampling steps: 20.
- Latent ottimizzato al primo o secondo timestep.
- Guidance scale: 1.
- Augmentation Kornia: vertical/horizontal flip, center crop, rotazione 90, channel dropout.
- Ensemble loss: somma a pesi uguali sui classifier disponibili.
- Raggio da 0.05 a 0.3, step 0.02.

Interpretazione:

MR-CAS mostra che manipolare lo spazio latente puo generare modifiche piu strutturate e meno simili a rumore fragile. La perturbazione non e solo una maschera pixel-level, ma una piccola riscrittura della traiettoria di ricostruzione. Questo puo aiutare la trasferibilita, perche modifica caratteristiche piu "alte" dell'immagine. Il prezzo e una SSIM media piu bassa: 0.742 contro 0.915/0.934 dei team 2 e 3.

Punto debole:

E computazionalmente costoso e potenzialmente rischioso con metriche percettive piu ricche. Nel 2026 entra LPIPS nello scoring, quindi dobbiamo verificare se le modifiche latenti migliorano davvero la similarita percettiva o se introducono cambi semantici/texture penalizzati.

### 2. Safe AI / MIG-COW: gradient ensemble con decomposizione consenso-ortogonale

Safe AI ha ottenuto lo score piu vicino a MR-CAS, con SSIM molto piu alta ma ASS piu basso. La loro idea e piu "classica" nel senso adversarial, ma molto elegante: produrre una direzione di update che combini vulnerabilita comuni e vulnerabilita specifiche dei modelli.

Elementi chiave:

- usa Integrated Gradients invece del gradiente diretto;
- il baseline e un'immagine nera;
- integra un termine di momentum;
- calcola le mappe di attribuzione per ogni modello sorgente;
- costruisce un gradiente di consenso come media dei gradienti normalizzati;
- costruisce una componente ortogonale tramite Gram matrix dei gradienti e autovettore associato all'autovalore minimo;
- rimuove la componente ridondante rispetto al consenso;
- combina consenso e ortogonale con peso beta;
- usa cross-entropy sui logit/target real, non probabilita raw.

Impostazioni riportate:

- epsilon: 0.02.
- passi: 25.
- momentum: 1.
- beta migliore: circa 0.7-0.8.

Risultati e insight:

- White-box ASR: 99.96%.
- Black-box ASR ufficiale: solo 7.16%, pur con buon punteggio complessivo.
- Aggiungere un modello ViT-P fra i sorgenti puo peggiorare il transfer black-box, anche se architetturalmente simile a un target blind.

Questa e forse la lezione piu importante del paper: non basta scegliere surrogate "diversi" o "simili per architettura". Conta la similarita funzionale: training data, obiettivo, preprocess, calibrazione e pattern decisionali. Un ViT debole o addestrato su distribuzione diversa puo sporcare il gradiente ensemble.

Punto forte per noi:

MIG-COW e relativamente implementabile e si adatta bene all'idea di ensemble 2026. Inoltre il concetto consenso + ortogonale puo essere esteso includendo trasformazioni di compressione e DCT, non solo modelli.

Punto debole:

Le perturbazioni restano pixel-space e potrebbero essere fragili sotto JPEG/social-media processing se non allenate esplicitamente con expectation over transformations.

### 3. RoMa: rumore globale distribuito e surrogate training

RoMa e il caso piu istruttivo per capire cosa non basta. Ha avuto SSIM altissima e quasi perfetto successo sui modelli pubblici, ma transfer pessimo sui blind.

Approccio:

- confronta tre famiglie: post-processing globale, patch localizzate, rumore adversarial distribuito;
- scarta post-processing perche abbassa SSIM e i detector pubblici sembrano robusti ad augmentations comuni;
- scarta patch perche devono coprire aree troppo grandi, danneggiando SSIM;
- sceglie rumore globale distribuito;
- addestra due surrogate: ViT-B16 ed EfficientNet-B0, usando immagini challenge piu FFHQ;
- distingue LQ e HQ:
  - LQ: rumore piccolo, ottimizzato con Adam, batch da 10, lr 0.0005;
  - HQ: DI-FGSM con resize/padding casuale e ottimizzazione per immagine.

Risultati:

- ResNet50 ASR: 0.9857.
- DenseNet121 ASR: 0.9800.
- ViT-B16 blind ASR: 0.0321.
- DenseNet121-DCT blind ASR: 0.0385.
- SSIM media: 0.9338.
- Final score: 2679.

Interpretazione:

RoMa dimostra che lo score 2025 poteva ancora essere alto anche con transfer black-box molto basso, se il metodo era quasi perfetto sui modelli pubblici e manteneva SSIM alta. Nel 2026 questo scenario e meno sicuro, perche i modelli pubblici sono gia ViT-B16 e DenseNet-DCT, e la valutazione finale include processing/compressione e probabilmente piu classifier.

Punto utile:

La separazione per risoluzione e molto sensata. LQ/HQ non sono lo stesso problema: le LQ hanno meno budget percettivo e meno informazione; le HQ permettono perturbazioni piu distribuite ma rischiano di essere ridimensionate/compresse prima della valutazione.

## Pattern comuni nella classifica

### Cosa ha funzionato

1. Ensemble di modelli.
   I metodi migliori non ottimizzano un solo detector. Anche quando il transfer resta difficile, l'ensemble riduce overfitting al singolo gradiente.

2. Momentum e trasformazioni.
   MI-FGSM, DI-FGSM, augmentation e input diversity ricorrono in molti team forti. Stabilizzano la direzione del gradiente e riducono fragilita.

3. Ottimizzazione score-aware.
   I team efficaci bilanciano esplicitamente loss di attacco e qualita. Dove manca il bilanciamento, uno dei due lati collassa.

4. Diversificazione spaziale/frequenziale.
   La presenza di DenseNet-DCT rende utile ragionare anche in luminanza, frequenza e low-frequency robustness.

5. Strategie per sottogruppi.
   HQ e LQ, generatori GAN e diffusion, immagini facili e difficili: trattarli ugualmente e subottimale.

### Cosa non ha funzionato abbastanza

1. Post-processing puro.
   Blur, compressione, warping o noise generico possono abbassare accuracy, ma spesso distruggono SSIM o sono gia coperti da data augmentation dei detector.

2. Patch localizzate.
   Troppo visibili o troppo penalizzanti per SSIM, a meno che la challenge non permetta alterazioni palesi.

3. Surrogate scelti solo per architettura.
   ViT contro ViT non basta: se il training non produce decision boundary simili, il transfer resta basso.

4. SSIM altissima senza attacco.
   MILab e il caso piu netto: preservare quasi perfettamente l'immagine non serve se i classifier restano su fake.

5. ASR alta con qualita troppo bassa.
   VYAKRITI 2.0 mostra l'estremo opposto: se la qualita collassa, il punteggio collassa.

## Implicazioni per AADD-2026

Le complicazioni 2026 cambiano parecchio il valore delle lezioni 2025.

Differenze operative gia note dal rulebook locale:

- test set 2026: 1600 immagini;
- modelli rilasciati: ViT-B/16 RGB e DenseNet121-DCT;
- scoring locale: `0.5 * SSIM + 0.5 * (1 - LPIPS)`, moltiplicato per il successo dell'attacco;
- la valutazione finale include compressioni JPEG/JPEG-AI-like e social-media-like processing;
- i modelli finali non coincidono necessariamente con i due pubblici;
- le fonti parlano di un ensemble finale piu ampio, probabilmente fino a 8 classifier, con una piccola ambiguita fra 6 e 8 in alcune fonti.

Conseguenze:

1. Il caso RoMa non basta piu.
   Fare quasi 100% sui pubblici e SSIM alta potrebbe non bastare, perche i pubblici 2026 sono gia piu difficili e perche la compressione puo cancellare perturbazioni fragili.

2. Il DCT non e piu blind.
   Quest'anno possiamo ottimizzare direttamente contro un detector frequenziale. Questo apre spazio per attacchi misti RGB + frequenza, ma aumenta anche il rischio di overfitting al preprocess esatto.

3. LPIPS cambia il compromesso.
   Nel 2025 bastava SSIM. Nel 2026 un attacco che conserva struttura ma altera texture percettive puo perdere punti. La loss deve includere un proxy LPIPS o almeno una selezione candidati basata su LPIPS reale.

4. Serve EOT con processing.
   Se gli organizzatori applicano JPEG/resize/social-like processing, dobbiamo ottimizzare sotto trasformazioni simulate. Una perturbazione non testata dopo re-encoding rischia di essere cancellata.

5. La trasferibilita deve essere misurata in modo piu severo.
   Non basta ASR sui due modelli locali. Serve un mini-benchmark interno con surrogate diversi e trasformazioni, anche se imperfetto.

## Direzione metodologica consigliata

La direzione piu promettente non sembra copiare uno dei tre metodi, ma costruire un ibrido pragmatico:

### Fase 1: baseline forte pixel/frequency/EOT

Implementare un attacco iterativo target `real` contro i due modelli pubblici 2026:

- ViT-B/16 RGB con preprocess ufficiale;
- DenseNet121-DCT con grayscale, resize/crop, DCT e log amplitude;
- loss target real su logits;
- penalita di similarita con SSIM e LPIPS o proxy LPIPS;
- EOT con JPEG quality variabile, resize, crop/center crop, blur leggero, re-encoding;
- momentum e input diversity;
- perturbazione vincolata e possibilmente low-frequency-biased per sopravvivere a compressione.

Questa fase serve a ottenere un baseline solido e misurabile.

### Fase 2: gradient ensemble in stile MIG-COW

Estendere la baseline con:

- Integrated Gradients o una variante semplificata per modelli sorgente;
- decomposizione consenso/ortogonale;
- selezione dei surrogate per performance funzionale, non solo architettura;
- confronto beta, epsilon e numero step usando score locale e score post-compression.

La domanda sperimentale chiave: COW aiuta anche quando uno dei modelli e DCT e quando i gradienti arrivano da trasformazioni EOT?

### Fase 3: candidate generation e selezione per immagine

Per ogni immagine generare piu candidati:

- attacco conservativo ad alta similarita;
- attacco piu aggressivo;
- variante low-frequency/DCT-aware;
- variante EOT-heavy;
- eventualmente variante latente per immagini difficili.

Poi selezionare il candidato migliore secondo uno score interno:

```text
score_atteso = media_su_trasformazioni_e_surrogate(similarity * indicator_real)
```

Questa parte e molto importante perche la formula ufficiale e discreta: se un candidato perde un classifier, il suo contributo su quel classifier va a zero. Conviene scegliere per immagine, non fissare un unico epsilon globale.

### Fase 4: opzione latente selettiva, non come primo passo

MR-CAS e molto interessante, ma usarlo come primo metodo rischia di essere lento e difficile da controllare. Una scelta piu robusta:

- prima pixel/EOT/MIG-COW per tutto il dataset;
- identificare immagini che restano fake su uno dei due pubblici o collassano dopo compressione;
- provare DDIM inversion/latent optimization solo su quel sottoinsieme;
- accettare il risultato latente solo se LPIPS e robustezza post-compression migliorano.

## Esperimenti prioritari

1. Baseline no-op.
   Valutare il test originale contro i due modelli pubblici: serve sapere quanto margine c'e e se tutte le immagini partono da `Fake`.

2. MI-FGSM/DI-FGSM target real.
   Versione semplice contro entrambi i pubblici, con epsilon sweep e misura SSIM/LPIPS.

3. DCT-aware attack.
   Verificare se ottimizzare direttamente nel flusso DCT porta perturbazioni visibili o gestibili.

4. EOT JPEG.
   Testare ogni candidato dopo griglia di compressione: quality 95, 90, 85, 80, 75, magari anche resize 0.75/0.5 e ritorno.

5. Surrogate zoo piccolo.
   Addestrare o usare detector aggiuntivi: ResNet/DenseNet RGB, EfficientNet, ViT/CLIP-like, eventualmente un secondo DCT detector. Misurare non solo architettura, ma accordo funzionale con i pubblici.

6. Per-image adaptive epsilon.
   Stimare il minimo epsilon che rende l'immagine `Real` e fermarsi appena il guadagno marginale non compensa il calo di similarita.

7. Candidate selection.
   Generare 3-5 varianti per immagine e scegliere con score interno post-processing-aware.

## Rischi principali

1. Overfitting ai due modelli pubblici.
   E il rischio piu grande. Nel 2025 molti metodi lo hanno fatto senza accorgersene fino alla fase finale.

2. Perturbazioni cancellate dalla compressione.
   Gli attacchi high-frequency pixel-space possono sembrare forti localmente e sparire dopo JPEG.

3. LPIPS sottovalutato.
   Un attacco con SSIM alta puo peggiorare LPIPS, soprattutto se cambia texture o volto in modo percettivamente rilevante.

4. Surrogate rumorosi.
   Aggiungere un modello debole puo peggiorare il gradiente, come mostrato da MIG-COW con ViT-P.

5. Costi computazionali.
   Latent diffusion e multi-candidate EOT su 1600 immagini puo diventare pesante. Serve una pipeline a budget progressivo.

## Takeaway operativo

Per il nostro approccio 2026 conviene partire da una pipeline score-aware e robusta a trasformazioni, non da un singolo attacco elegante. L'architettura mentale migliore e:

```text
public-model success
  + perceptual similarity SSIM/LPIPS
  + robustness to JPEG/social processing
  + transfer to surrogate family
  + per-image candidate selection
```

Fra i metodi 2025, MIG-COW e probabilmente il miglior punto di partenza ingegneristico; MR-CAS e la fonte di idee piu originale per un secondo livello latente; RoMa e il warning principale: SSIM alta e white-box quasi perfetto possono nascondere un transfer molto fragile.

