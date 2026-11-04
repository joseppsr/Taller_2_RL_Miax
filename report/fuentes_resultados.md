# Fuentes del informe actualizado

Documento de trazabilidad; no forma parte del PDF de dos páginas.

| Contenido del informe | Archivos de resultados |
|---|---|
| Test principal, cuatro mercados y agentes | `results/tabla.csv`, `results/test_por_serie.csv.gz`, `results/test_baselines.csv` |
| Configuraciones originales | `results/mejores.csv` |
| MSE/MAE, minibatch y episodio entero | `results/ajuste_dqn_c.csv`, `results/ajuste_a3c_c.csv`, `results/lotes_test.csv.gz` |
| Concurrencia y convergencia de A3C | `results/a3c_sin_cerrojo.csv`, `results/escalado_a3c.csv`, `results/convergencia_a3c.csv` |
| Datos reales | `results/reales_resultados.csv` |
| Búsqueda inicial de estabilidad DQN, semillas 350–352 | `results/estabilidad_dqn_reversion/6c28a6b8b526/` |
| Confirmación DQN, semillas 360–364 | `results/estabilidad_dqn_reversion/489842ba036f/` |
| A3C: cuatro variantes, 12.800 episodios | `results/estabilidad_a3c_reversion/286127b6167a/` |
| A3C: tres calendarios, 3.200 episodios | `results/estabilidad_a3c_reversion/b524af4c8dbb/` |
| A3C original: cinco semillas y test reservado | `results/estabilidad_a3c_reversion/1e9c1784f2f4/decision_66330b4a66cf.json`, `decision_66330b4a66cf_test.csv` en esa carpeta |
| Repetición de hiperparámetros con cinco semillas | `results/ajuste_robusto/e48237edf3e5/comparacion.csv`, `propuestas.csv` en esa carpeta |

También se revisaron las repeticiones cortas de A3C en `0a15192dd759`, `e35db19ef62e`, `fe012d78fb31` y `91760e1985d9`. Sus conclusiones se resumen como sensibilidad al calendario y a la ejecución; no se mezclan sus medias con la búsqueda posterior de tres calendarios.

Las cifras de validación de estabilidad promedian las métricas por semilla. Las rentabilidades del test principal son las medianas agregadas de su tabla original. El test reservado de A3C informa la media entre semillas de las medianas por serie. No se interpretan como estimadores idénticos.

El experimento de hiperparámetros utiliza los algoritmos originales: sus conclusiones no sustituyen la confirmación de DQN con aprendizaje decreciente. Esta última aún no tiene un test reservado guardado en el proyecto.

## Regenerar el PDF

```powershell
.\.venv\Scripts\python -m pip install -r report/requirements.txt
.\.venv\Scripts\python report/generar_informe.py
```

El script lee `report/informe.md`, comprueba que haya exactamente dos páginas y genera imágenes de revisión en `report/preview/`. La versión anterior se conserva en `report/informe_anterior.pdf` y `report/informe_anterior.md`.
