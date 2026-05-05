# AADD-2026 Rulebook Operativo

Ultimo controllo locale: 2026-05-05.

## Fonti analizzate

- Email ufficiale AADD-2026 incollata nella chat.
- `/Users/anto/AADD_Challenge/AADD_2026_config.yaml`
- `/Users/anto/AADD_Challenge/AADD_2026_evaluation.py`
- `/Users/anto/AADD_Challenge/Ricerca/AAAD_Challenge_2026_Adversar.md`
- Sito ufficiale: https://iplab.dmi.unict.it/mfs/acm-aadd-challenge-2026/
- OpenReview: https://openreview.net/forum?id=G0I7DYvzrp

## Obiettivo

Generare una versione adversarial dell'intero test set di immagini deepfake in modo che i detector le predicano come classe `Real`, preservando la qualita visiva.

La classe da colpire e:

- `Real` = indice `0`
- `Fake` = indice `1`

Nel codice ufficiale un attacco riuscito vale solo quando:

```text
pred == CLASS_IDX_REAL
```

## Materiali ufficiali

- Test set: `AADD_2026_Test`, 1600 immagini.
- Pesi rilasciati:
  - `vit_b_16.pth`
  - `densenet121_dct.pth`
- Script di valutazione: `AADD_2026_evaluation.py`
- Configurazione: `AADD_2026_config.yaml`

I due modelli rilasciati sono solo riferimento di sviluppo. La valutazione finale avviene su modelli non completamente visibili ai partecipanti. Le fonti parlano di 8 classificatori totali, ma il sito contiene anche una riga che menziona 6 classificatori nella fase dopo post-processing: trattare questo come ambiguita, non come garanzia.

## Dati locali osservati

- Numero immagini: 1600.
- Tutti i nomi hanno estensione `.png`.
- Struttura: piatta, senza sottocartelle.
- Esempi nomi: `000.png`, `001.png`, ..., fino a nomi oltre `2024.png`; non assumere progressione continua.
- Modalita: RGB per tutte le immagini.
- Dimensioni presenti:
  - 512x512: 164
  - 640x640: 61
  - 768x768: 50
  - 1024x1024: 1259
  - 1792x1792: 66
- Formato reale letto da PIL:
  - PNG: 1387
  - JPEG: 213

Nota critica: 213 file hanno nome `.png` ma contenuto JPEG. Per la challenge conta soprattutto che i nomi restino identici e che PIL possa aprire le immagini. Evitare qualunque rinomina.

## Formula di scoring nello script

Per ogni coppia originale/adversarial:

```text
similarity = alpha * SSIM + (1 - alpha) * (1 - LPIPS)
```

con:

```text
alpha = 0.5
```

Per ogni classificatore:

```text
contribution = classifier_weight * similarity * indicator
```

dove:

```text
indicator = 1 se il modello predice Real, altrimenti 0
```

Con la config ufficiale locale:

```text
aggregate: sum
```

quindi lo script somma i contributi su immagini e classificatori. Se tutte le immagini ingannano entrambi i modelli rilasciati con similarity circa 1, il massimo locale con 2 classifier e circa 3200.

## Trasformazioni dei modelli rilasciati

### `vit_b_16`

- Input RGB.
- Resize a 256x256.
- Center crop 224x224.
- Normalizzazione ImageNet:
  - mean = `[0.485, 0.456, 0.406]`
  - std = `[0.229, 0.224, 0.225]`

Implicazione: il centro dell'immagine e molto importante; perturbazioni solo ai bordi potrebbero non essere viste dal ViT.

### `densenet121_dct`

- Conversione in grayscale.
- Se dimensione massima > 256, resize a 256x256.
- Center crop 128x128.
- DCT 2D.
- Log scale: `log(abs(dct) + 1e-6)`.
- Input 1 canale.

Implicazione: serve robustezza anche in frequenza e in luminanza, non solo pattern RGB spaziali.

## Valutazione finale e rischio black-box

La valutazione ufficiale non coincide necessariamente con il solo script locale:

- I due modelli rilasciati sono reference, non l'intero ensemble finale.
- Il PDF e il sito insistono su JPEG/JPEG-AI compression e social-media-like processing.
- Il sito dice che il test set attaccato verra modellato casualmente tramite compressioni e processing prima della valutazione.
- Serve quindi un attacco trasferibile e robusto a compressione/resize/re-encoding, non una perturbazione fragile ottimizzata solo sui due pesi.

## Config: cosa modificare e cosa non modificare

Da modificare:

- `original_root`: cartella test originale.
- `adv_root`: cartella immagini adversarial.
- `models_dir`: cartella con i `.pth`.
- `save_json`: file JSON di output, non directory.

Da non modificare per il check ufficiale:

- `classifiers`
- `dct_log_scale`
- `weights`
- `aggregate`
- `device`, salvo necessita pratica CPU/GPU
- `alpha`

Problemi nella config locale attuale:

- `adv_root` punta ancora al test originale: va cambiato quando esiste una cartella adversarial.
- `models_dir` ha `//Users/...`: probabilmente funziona su macOS, ma meglio usare `/Users/...`.
- `save_json` punta a una directory (`/Users/anto/AADD_Challenge/results`), mentre lo script apre un file. Usare per esempio `/Users/anto/AADD_Challenge/results/eval.json`.

## Regole di consegna

Inviare via email a:

```text
challenge.dff@gmail.com
```

Oggetto consigliato:

```text
AADD-2026 Challenge Submission - [Team Name]
```

Materiali richiesti:

- ZIP adversarial: `AADD-2026_Adversarial_Test_Set_<TEAM-NAME>.zip`
- Abstract PDF di 1-2 pagine con metodologia, motivazione e contributi.
- Note brevi opzionali nel corpo email.

Regole per lo ZIP:

- Deve contenere tutte le 1600 immagini adversarial.
- I nomi devono essere identici agli originali.
- Non cambiare ID, formato nome o struttura.
- Escludere file estranei come `.DS_Store`, `__MACOSX`, log, JSON, notebook.
- Poiche il test locale e piatto, lo ZIP piu sicuro e con le immagini direttamente alla radice oppure con struttura esattamente equivalente a quella originale se gli organizzatori l'hanno richiesta cosi.

## Date ufficiali rilevanti

Dal sito ufficiale:

- Registrazione: 17 marzo - 18 maggio 2026.
- Release test set e classificatori: 20 aprile 2026.
- Submission results: 10 giugno 2026.
- Leaderboard publication: 15 giugno 2026.
- Final paper top 3 only: 25 giugno 2026.
- Paper decision: 16 luglio 2026.
- Camera ready top 3 only: 6 agosto 2026.

## Checklist pre-submission

Prima di inviare:

1. La cartella adversarial contiene esattamente 1600 immagini.
2. Ogni nome corrisponde esattamente a un nome del test set.
3. Non ci sono file mancanti o extra.
4. PIL apre tutte le immagini.
5. Le immagini sono RGB o convertibili correttamente in RGB.
6. La valutazione ufficiale gira senza eccezioni.
7. `save_json` e un file `.json`, non una directory.
8. Il risultato e controllato sia su `vit_b_16` sia su `densenet121_dct`.
9. L'attacco e testato anche dopo JPEG compression e resize.
10. Lo ZIP non contiene metadati macOS o cartelle non richieste.

## Priorita strategiche

Ordine consigliato:

1. Prima massimizzare il tasso di predizione `Real` su entrambi i modelli rilasciati.
2. Poi mantenere alta similarity: SSIM alto e LPIPS basso.
3. Poi testare robustezza a JPEG/JPEG-AI-like compression, resize e re-encoding.
4. Infine validare trasferibilita con modelli surrogate diversi, per non sovradattarsi ai due detector disponibili.

