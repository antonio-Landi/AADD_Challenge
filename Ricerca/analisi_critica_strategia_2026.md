# Analisi critica e strategia per AADD-2026

Data: 2026-05-05.

---

## 1. Problemi nelle fonti citate

### MIG-COW: la trasferibilità è quasi zero, non buona

Il paper riporta 99.96% white-box ASR e lo usa come titolo. Il dato operativo è 7.16% black-box ASR. Non è "una sfida", è un fallimento del trasferimento. Il vantaggio in classifica deriva dall'SSIM alta (0.915) su un numero enorme di immagini dove i modelli pubblici 2025 (ResNet50, DenseNet121) erano facili da battere. La decomposizione consenso/ortogonale è matematicamente coerente ma non ha alcuna garanzia teorica di migliorare il transfer verso modelli con architettura, training data e decision boundary diversi. L'unico supporto empirico per "functional similarity > architectural similarity" è una singola osservazione (ViT-P peggiora il transfer) con una spiegazione alternativa ovvia: ViT-P è un modello di qualità inferiore (~70% accuracy) che inquina il gradiente. Non è evidenza di un principio generale.

**In AADD-2026, MIG-COW parte svantaggiato**: i due modelli pubblici 2026 sono esattamente i blind 2025 che MIG-COW non riusciva a raggiungere. Il 7.16% BB-ASR è il numero rilevante per 2026, non il 99.96%.

### MR-CAS: il paper non riporta LPIPS

L'analisi di AADD-2025 non include LPIPS perché lo scoring 2025 non lo usava. Non sappiamo se la DDIM inversion introduca cambi percettivi penalizzati da LPIPS. Latent space manipulation può:

- Cambiare identità o espressione del volto → LPIPS alto
- Introdurre texture sintetiche → LPIPS alto
- Modificare il colore dominante della scena → SSIM OK, LPIPS pessimo

Il costo computazionale (SD inference per 1600 immagini ad alta risoluzione) non è stato analizzato per il 2026 dove le immagini arrivano fino a 1792×1792.

### Paper Guarnera et al. (DCT gray-box): il target è sbagliato

L'attacco è progettato per classifier che usano 63 parametri β (scale di distribuzioni Laplaciane degli AC coefficient su blocchi 8×8). Il modello AADD-2026 `densenet121_dct` usa:

- DCT 2D globale su patch 128×128 (non blocchi 8×8)
- `log(|DCT| + 1e-6)`
- DenseNet121 che apprende features dal log-magnitude spectrum

Questi sono domini diversi. Copiare l'approccio Guarnera non funziona direttamente. Il paper viene citato come ispirazione per un "spectral regularizer" ma non c'è evidenza che allineare statistiche 8×8 Laplaciane aiuti contro una DenseNet che guarda lo spettro globale.

### EOT: l'assunzione sulla distribuzione è non falsificabile

Athalye et al. dimostrano che EOT funziona quando si modellizza la distribuzione di trasformazioni corretta. Per AADD-2026 non sappiamo:

- A quale quality factor applicheranno JPEG
- Se comprimono prima o dopo resize
- Se usano JPEG-AI o JPEG standard
- Se applicano social media color normalization

Ottimizzare su QF={95,90,85} è una scommessa. Se il QF reale è 60-70, si ottiene un attacco sub-ottimale in una zona sbagliata. L'ensemble QF dà copertura ma diluisce il gradiente, richiedendo epsilon più alto che danneggia SSIM/LPIPS.

### I documenti di analisi: circolarità e bias di conferma

`aadd_2025_retro_analysis.md` e `jpeg_dct_attack_research.md` sono sintesi che raccomandano MIG-COW come "baseline solida" basandosi sui risultati 2025 con condizioni di scoring diverse. La frase "MIG-COW è probabilmente il miglior punto di partenza ingegneristico" non ha supporto empirico per il setting 2026 dove:

1. I modelli pubblici sono più difficili (erano i blind del 2025)
2. LPIPS è nello score
3. JPEG è applicato prima della valutazione

---

## 2. Geometria dello scoring: cosa davvero conta

La formula 2026 per immagine `k` e classifier `c`:

```
score(k,c) = clf_weight × (0.5×SSIM + 0.5×(1−LPIPS)) × I[pred==Real]
```

Il termine di similitudine vale nel range [0,1] e si azzera se l'attacco fallisce. Questo crea tre regimi:

| Scenario | SSIM | LPIPS | ASR | Score/immagine/classifier |
|---|---|---|---|---|
| Alta qualità, basso transfer | 0.95 | 0.05 | 0.08 | 0.076 |
| Media qualità, buon transfer | 0.85 | 0.10 | 0.65 | 0.552 |
| Bassa qualità, alto transfer | 0.75 | 0.20 | 0.90 | 0.585 |

Il caso MILab (SSIM 0.994, ASR 0.020) ottiene 0.020 × 0.994 = 0.020 per immagine per classifier. Un attacco "aggressivo ma robusto" vince. **La conservazione visiva non ha valore intrinseco — vale solo se l'attacco riesce.**

Il corollario: per immagini "difficili" dove nessun metodo converge, è meglio spendere budget adversarial in modo più aggressivo. La selezione per immagine non riguarda "quale candidato ha SSIM più alta" ma "quale candidato massimizza (similarity × indicator)".

---

## 3. Confronto strategie principali

### Strategia A: MIG-COW puro (baseline attuale)

- **Pro**: Implementata, funziona sui modelli pubblici 2026.
- **Contro**: 7.16% BB-ASR in 2025 su modelli identici a quelli pubblici 2026. Nessuna JPEG-awareness. Lento (IG × models × steps). LPIPS non ottimizzata.
- **Rischio principale**: Fallisce quasi tutto sull'ensemble blind.

### Strategia B: PGD/MI-FGSM con EOT su JPEG + ensemble surrogate

- **Pro**: Approccio classico, ben compreso, testato.
- **Contro**: Risultati dipendono da quali surrogate si scelgono. RoMa ha fallito con questo approccio. La scelta del QF range per EOT è una scommessa.
- **Rischio principale**: Stesso problema di RoMa se i surrogate non matchano i blind.

### Strategia C: Latent diffusion (MR-CAS)

- **Pro**: Meglio del pixel-space in 2025, perturbazioni strutturalmente coerenti.
- **Contro**: Richiede Stable Diffusion, lento, LPIPS ignoto, radius 0.05→0.30 produce immagini visibilmente alterate per i casi difficili.
- **Rischio principale**: LPIPS peggiora l'efficacia se introduce artefatti semantici.

### Strategia D: Attacco frequenziale diretto

- **Pro**: Sfrutta il preprocess noto del DCT model, perturbazioni JPEG-stabili by design.
- **Contro**: Non direttamente applicabile al ViT RGB. Serve un meccanismo di coordinamento tra i due domini.
- **Potenziale**: Non esplorato nel contesto AADD-2026.

---

## 4. Direzioni originali

### Idea 1: Aliasing adversariale — sfruttare il downsampling come moltiplicatore

**Meccanismo**: I modelli AADD-2026 non vedono l'immagine originale. ViT-B/16 vede 224×224 al centro di una 256×256; DenseNet-DCT vede 128×128 al centro di una 256×256. Per un'immagine originale di 1024×1024, il resize 1024→256 introduce aliasing spettrale.

La proprietà cruciale: un segnale sinusoidale a frequenza `f` in un'immagine 1024×1024 può diventare un segnale a frequenza `f/4` dopo resize a 256. Quindi perturbazioni **ad alta frequenza nell'originale (invisibili all'occhio e a LPIPS)** possono creare segnali **a bassa/media frequenza nello spazio di input del modello**.

L'idea: progettare `δ` direttamente nel dominio frequenziale dell'immagine 1024×1024 tale che `downsample(x + δ, 256) = downsample(x, 256) + δ_target`, dove `δ_target` è una perturbazione adversariale ottimizzata per i modelli. In pratica:

```
δ = IFFT( FrequencyMask_alias × θ )
```

dove `FrequencyMask_alias` seleziona le frequenze in 1024-space che si mappano alle frequenze target in 256-space.

**Vantaggi**:
- `δ` è quasi invisibile nell'originale (alta frequenza, bassa ampiezza)
- SSIM e LPIPS sono calcolati sull'immagine originale → molto vicini a 1.0
- Il segnale adversariale è "costruito" nella risoluzione del modello, non trasferito

**Criticità**:
- L'aliasing di `F.interpolate` non è deterministico tra PyTorch bilinear e PIL LANCZOS
- Il calcolo deve matchare esattamente il downsampling dell'evaluation script
- Per immagini già piccole (512×512) l'aliasing space è limitato
- JPEG può cancellare le alte frequenze originali prima del resize

**Ipotesi di applicazione**: le immagini con risoluzione ≥768×768 (>1300 nel test set) sarebbero i candidati ideali. Per le 512×512 si usa un attacco tradizionale.

---

### Idea 2: Multi-scale perturbation con oracle locale — sfruttare la struttura del score

**Meccanismo**: La formula di score premia perturbazioni che simultaneamente sono simili all'originale e funzionano su entrambi i modelli pubblici, sopravvivendo a post-processing ignoto. Invece di ottimizzare un singolo attacco, costruire un **oracolo locale** che stima il contributo atteso per ogni candidato:

```python
expected_score(candidate) = 
    mean_over_QF [ similarity(orig, candidate) * 
                   mean_over_models [ indicator(compress(QF, candidate)) == Real ] ]
```

Generare candidati diversificati in modo sistematico:

- **Candidato A**: PGD puro in pixel space, epsilon piccolo → alta qualità
- **Candidato B**: PGD con Luma-only perturbation → JPEG-stabile, visivamente neutro
- **Candidato C**: Attacco tramite DCT differenziabile bilanciato tra ViT e DCT model
- **Candidato D**: PGD con frequenza filtrata (passa-basso su δ) → perturbazione naturale

Per ogni immagine, si calcola `expected_score` su {QF=85,90,95,no-JPEG} e si sceglie il candidato che massimizza il prodotto similarity×success medio.

**Vantaggio rispetto all'analisi esistente**: l'analisi dice "candidate selection" ma non specifica come generare candidati con diversità controllata. Questa proposta definisce esplicitamente le dimensioni di diversificazione (frequency spectrum, color channel, JPEG stability).

**Criticità**:
- Computazionalmente costoso (4 candidati × 4 QF × 2 modelli = 32 evaluations per immagine)
- La distribuzione QF usata per selection potrebbe non matchare quella finale
- Il "candidato migliore su QF={85,90,95}" potrebbe essere diverso dal "migliore su QF=70"

---

### Idea 3: Attacco tramite statistica del log-spettro — "spectral forgery matching"

**Meccanismo**: Il modello `densenet121_dct` vede `log(|DCT(gray_crop(x))| + 1e-6)`. Questa rappresentazione ha proprietà statistiche che differiscono sistematicamente tra immagini reali e sintetiche. La differenza più nota: le immagini reali hanno distribuzioni di energia spettrale più "1/f" (decrescente con la frequenza), mentre i generatori moderni producono anomalie a specifiche frequenze (spesso nella griglia 8×8 lasciata da convoluzioni).

L'idea: costruire per ogni immagine fake un **target spettrale** che si avvicini a quello di immagini reali, nel dominio del preprocess ufficiale AADD-2026 (DCT globale 128×128 su grayscale crop). Poi ottimizzare `x_adv` per:

```
L = L_ce_dct(x_adv, Real) + λ × ||profile_AADD(x_adv) - centroid_real||_2
```

dove:

```
profile_AADD(x) = radial_energy_bands( log(|DCT_128(gray_crop(x))| + 1e-6) )
centroid_real   = kmeans centroid da FFHQ images processate con lo stesso pipeline
```

La novità rispetto all'analisi esistente: il profilo corretto per AADD-2026 è il **log-magnitude spectrum 128×128 in grayscale**, non le statistiche Laplaciane 8×8 di Guarnera. Questo profilo è direttamente ottimizzabile poiché il preprocess è differenziabile (vedi `_DifferentiableDCTModel` in `mig_cow_attack.py`).

**Vantaggio per il blind ensemble**: se altri classifier nell'ensemble sconosciuto usano varianti di analisi frequenziale, un'immagine con spettro "real-like" nel dominio luma tende a ingannare anche loro senza gradient access.

**Criticità**:
- Richiede immagini reali per costruire i centroidi. Disponibili: FFHQ, CelebA-HQ, ma non sappiamo se i detector blind sono stati addestrati su quelle distribuzioni.
- Ottimizzare SSIM/LPIPS e il profilo spettrale contemporaneamente può creare conflitti: il profilo real richiede redistribuire l'energia spettrale, il che cambia la texture visibile.
- L'effetto su ViT-B/16 è indiretto e non garantito. ViT opera in pixel space e non "vede" lo spettro esplicitamente.
- Il matching spettrale potrebbe produrre immagini che ingannano il DCT model ma non il ViT, senza beneficio netto se entrambi devono essere ingannati per massimizzare il contributo.

---

## 5. Valutazione complessiva

Il rischio principale non è tecnico ma epistemico: non sappiamo cosa sono i blind classifier. Le raccomandazioni nella letteratura analizzata sono ragionevoli come scommesse, ma nessuna ha un supporto empirico forte nel setting AADD-2026 specifico.

La strategia più difendibile è quella che **minimizza la dipendenza dall'identità dei blind classifier**: perturbazioni che sfruttano proprietà strutturali condivise da tutti i detector deepfake (anomalie frequenziali, textures dei generatori, pattern spaziali), piuttosto che metodi che ottimizzano il gradiente su modelli proxy che potrebbero non corrispondere.

In questo senso:

- **Idea 3 (spectral forgery matching)** è la più robustamente motivata: attacca una vulnerabilità strutturale condivisa da tutti i detector frequenziali, non un modello specifico.
- **Idea 1 (aliasing)** ha il massimo potenziale su SSIM/LPIPS ma richiede validazione empirica sull'aliasing del preprocess ufficiale. Priorità alta per le immagini ad alta risoluzione (≥768px).
- **Idea 2 (multi-scale oracle)** è complementare alle altre due e richiede implementazione incrementale; può essere usata come layer di selezione finale.

### Gap principali nell'implementazione attuale (`mig_cow_attack.py`)

1. Nessuna JPEG-EOT durante l'ottimizzazione
2. Nessun profilo spettrale real come regularizer
3. Epsilon fisso per tutte le immagini (non adattivo per-image)
4. Nessuna candidate selection basata su score atteso post-compression
5. LPIPS non inclusa nella loss di attacco, solo nelle metriche di evaluation
