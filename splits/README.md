# Partición por grupos de fotos

Desde la raíz del repositorio y con el dataset en `data/dataset`, ejecutá `python 1-pre-processing.py` en el entorno del proyecto o corré `1-pre-processing.ipynb` de principio a fin. Se usa la semilla 42 para el reparto y para los aumentos. El primer segmento de una ruta como `valid/Afghan/05.jpg` indica su carpeta de Kaggle. La columna `split` identifica su partición final.

## Evidencia y decisiones

- El registro conserva los 405 pares de la auditoría histórica, incluidos 396 confirmados en 296 grupos y nueve descartados. Se recuperaron de las tablas guardadas en `1-pre-processing.ipynb` del commit `4187cc43d671320cbebd004b86083a1db509caf9` y se verificó su firma conjunta `7ef8c29886bbb98a6516df3f5c61f2b7b6fda82447f38e13f9c4533bd096ab0c` antes de regenerar los CSV. Las rondas siguientes agregaron decisiones con las mismas claves de ruta y los SHA-256 de las dos fotos. [Registro de pares](audit_pairs.csv)
- Se documentaron las etiquetas discordantes, todas las rutas de cada grupo, cuántos originales se excluyeron y el motivo. Se excluyó cada grupo completo porque la correspondencia visual identifica la foto, y las carpetas ofrecen etiquetas contradictorias sin una fuente verificable para elegir una. Las copias de una foto comparten origen, así que una mayoría de carpetas puede repetir el mismo error. [Registro de conflictos](audit_conflicts.csv)

La auditoría cruza los originales de `train`, `valid` y `test` con SHA-256, correspondencias locales SIFT y similitud global entre recortes. Si aparecen pares sin decisión, el proceso escribe `output/audit_pending.csv` y detiene la publicación. Compará las fotos y agregá la decisión y el motivo a `audit_pairs.csv` con `path_a < path_b` en orden lexicográfico y los hashes en el mismo orden. Los pares `confirmed` unen las componentes antes del nuevo reparto. Volvé a ejecutar el proceso para revisar los cruces del reparto actualizado. El resultado cubre las coincidencias confirmadas por este procedimiento.

## Resultado

| Split | Originales | Aumentos | Imágenes | Razas |
| --- | ---: | ---: | ---: | ---: |
| `train` | 6.393 | 4.107 | 10.500 | 70 |
| `valid` | 1.371 | 0 | 1.371 | 70 |
| `test` | 1.371 | 0 | 1.371 | 70 |
| Total | 9.135 | 4.107 | 13.242 | 70 |

Las rondas posteriores a la auditoría histórica incorporaron 309 pares confirmados y cuatro descartes. La auditoría del reparto final dejó cero grupos confirmados en más de un split y cero candidatos pendientes. Se excluyeron 22 grupos con etiquetas discordantes, que reunían 50 originales que habían pasado el filtro de calidad. Los 4.107 aumentos se generaron desde los originales de `train`, con 150 imágenes por raza después de equilibrar.

## Checkpoints de la partición anterior

Los cuatro checkpoints de clasificación existentes se conservaron en sus rutas originales. Fueron entrenados con la partición anterior, cuyos manifiestos están en el commit `4187cc43d671320cbebd004b86083a1db509caf9`. Las métricas y las salidas guardadas de `2-train.ipynb` son históricas. Para el próximo entrenamiento configurá otros valores de `MODEL_PATH` y `OUTPUT_PATH`.

| Archivo relativo a `models/` | SHA-256 |
| --- | --- |
| `cnn_custom.pth` | `d823616a4a26ef579b99866c8db967ab05aac806c3eb2318f6899c93e81d8e90` |
| `resnet18_finetuned.pth` | `6bdd648dfbe8211ab15e5c082fdcd8a04bf4989851d0c0ca703f6d2550eeadfa` |
| `cuda-prueba-20261126-212552/cnn_custom.pth` | `f47713f1a993e0737ed457bf61bffba66b636f9ffa69c71909af246e2ebf28ab` |
| `cuda-prueba-20261126-212552/resnet18_finetuned.pth` | `7891cca7ff3f86200762756ab1877364666d3a8a77ae8eea78701a6b02290e17` |
