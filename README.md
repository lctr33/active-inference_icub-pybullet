# Active Inference iCub PyBullet

Proyecto experimental para implementar inferencia activa como POMDP en un agente iCub simulado en PyBullet.
El entorno utiliza el modelo `iCubGazeboV2_5_visuomanip` de `icub-models`, una discretización del espacio frente al rostro y renderizado visual desde la cámara del robot.

## Requisitos generales

* Git
* Miniforge / Conda
* Python 3.10 o 3.11
* `icub-models`
* PyBullet
* NumPy
* OpenCV

El proyecto usa la variable de entorno `ICUB_URDF` para localizar el modelo URDF del iCub.

---

# Instalación en Ubuntu

## 1. Instalar dependencias del sistema

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

## 2. Instalar Miniforge

Descarga el instalador:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

Ejecuta la instalación:

```bash
bash Miniforge3-Linux-x86_64.sh
```

Cierra y vuelve a abrir la terminal, o ejecuta:

```bash
source ~/miniforge3/bin/activate
conda init bash
```

Después reinicia la terminal.

Verifica la instalación:

```bash
conda --version
mamba --version
```

## 3. Clonar el repositorio

```bash
git clone https://github.com/lctr33/active-inference_icub-pybullet
cd active-inference_icub-pybullet
```

## 4. Crear el entorno Conda

```bash
mamba create -n icub-pybullet python=3.11 -y
conda activate icub-pybullet
```

## 5. Instalar `icub-models`

```bash
mamba install -c conda-forge icub-models -y
```

## 6. Instalar dependencias de Python

```bash
pip install numpy opencv-python pybullet
```

## 7. Ubicar el modelo `iCubGazeboV2_5_visuomanip`

Busca el archivo `model.urdf`:

```bash
find "$CONDA_PREFIX" -path "*iCubGazeboV2_5_visuomanip/model.urdf"
```

Normalmente estará en una ruta similar a:

```bash
$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model.urdf
```

Por ejemplo:

```bash
/home/usuario/miniforge3/envs/icub-pybullet/share/iCub/robots/iCubGazeboV2_5_visuomanip/model.urdf
```

## 8. Definir la variable `ICUB_URDF`

Usa la ruta real encontrada en el paso anterior:

```bash
export ICUB_URDF="$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model.urdf"
```

Para hacerla permanente:

```bash
echo 'export ICUB_URDF="$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model.urdf"' >> ~/.bashrc
source ~/.bashrc
```

Verifica:

```bash
echo "$ICUB_URDF"
```

---

# Parcheo de dummy inertials en Ubuntu

Algunas versiones del modelo iCub incluyen enlaces con inercias dummy que pueden generar advertencias o problemas al cargar el URDF en PyBullet.
Antes de ejecutar la simulación, se debe generar una versión parcheada del modelo.

## 1. Abrir el script de parcheo

Edita el script de parcheo incluido en el proyecto. Por ejemplo:

```bash
nano scripts/patch_dummy_inertials.py
```

Dentro del script, coloca la ruta del modelo:

```python
ICUB_URDF = "/home/usuario/miniforge3/envs/icub-pybullet/share/iCub/robots/iCubGazeboV2_5_visuomanip/model.urdf"
```

También puede usarse la variable de entorno:

```python
import os
ICUB_URDF = os.environ["ICUB_URDF"]
```

## 2. Ejecutar el parcheo

```bash
python scripts/patch_dummy_inertials.py
```

El script debe generar un archivo parcheado en la misma carpeta del modelo, por ejemplo:

```bash
$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model_patched.urdf
```

Mantener el URDF parcheado dentro de la misma carpeta del modelo evita romper rutas relativas hacia meshes y materiales.

## 3. Actualizar `ICUB_URDF` para usar el modelo parcheado

```bash
export ICUB_URDF="$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model_patched.urdf"
```

Para hacerlo permanente:

```bash
echo 'export ICUB_URDF="$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model_patched.urdf"' >> ~/.bashrc
source ~/.bashrc
```

---

# Instalación en Windows

Se recomienda usar PowerShell o Miniforge Prompt.

## 1. Instalar Git

Descarga e instala Git para Windows:

```text
https://git-scm.com/download/win
```

Durante la instalación puedes dejar las opciones por defecto.

Verifica:

```powershell
git --version
```

## 2. Instalar Miniforge

Descarga el instalador de Windows x86_64 desde:

```text
https://github.com/conda-forge/miniforge/releases/latest
```

Archivo esperado:

```text
Miniforge3-Windows-x86_64.exe
```

Instálalo y abre **Miniforge Prompt** desde el menú de inicio.

Verifica:

```powershell
conda --version
mamba --version
```

## 3. Clonar el repositorio

```powershell
git clone clone https://github.com/lctr33/active-inference_icub-pybullet
cd active-inference_icub-pybullet
```

## 4. Crear el entorno Conda

```powershell
mamba create -n icub-pybullet python=3.11 -y
conda activate icub-pybullet
```

## 5. Instalar `icub-models`

```powershell
mamba install -c conda-forge icub-models -y
```

## 6. Instalar dependencias de Python

```powershell
pip install numpy opencv-python pybullet
```

## 7. Ubicar el modelo `iCubGazeboV2_5_visuomanip`

En Windows, los archivos instalados por Conda suelen quedar dentro de `Library`.

Ejecuta:

```powershell
Get-ChildItem -Path $env:CONDA_PREFIX -Recurse -Filter model.urdf |
Where-Object { $_.FullName -like "*iCubGazeboV2_5_visuomanip*" } |
Select-Object -ExpandProperty FullName
```

La ruta normalmente será similar a:

```powershell
C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model.urdf
```

## 8. Definir la variable `ICUB_URDF`

Temporalmente, solo para la terminal actual:

```powershell
$env:ICUB_URDF="C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model.urdf"
```

Para guardarla de forma permanente:

```powershell
setx ICUB_URDF "C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model.urdf"
```

Después de usar `setx`, cierra y vuelve a abrir Miniforge Prompt.

Verifica:

```powershell
echo $env:ICUB_URDF
```

---

# Parcheo de dummy inertials en Windows

## 1. Abrir el script de parcheo

Edita el script:

```powershell
notepad scripts\patch_dummy_inertials.py
```

Coloca la ruta del modelo original:

```python
ICUB_URDF = r"C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model.urdf"
```

También puede usarse la variable de entorno:

```python
import os
ICUB_URDF = os.environ["ICUB_URDF"]
```

## 2. Ejecutar el parcheo

```powershell
python scripts\patch_dummy_inertials.py
```

El script debe generar un archivo como:

```powershell
C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model_patched.urdf
```

## 3. Actualizar `ICUB_URDF` para usar el modelo parcheado

Temporalmente:

```powershell
$env:ICUB_URDF="C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model_patched.urdf"
```

Permanentemente:

```powershell
setx ICUB_URDF "C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model_patched.urdf"
```

Reabre Miniforge Prompt después de usar `setx`.

---

# Ejecución del proyecto

Activa el entorno:

## Ubuntu

```bash
conda activate icub-pybullet
```

## Windows

```powershell
conda activate icub-pybullet
```

Verifica que `ICUB_URDF` apunta al modelo parcheado:

## Ubuntu

```bash
echo "$ICUB_URDF"
```

## Windows

```powershell
echo $env:ICUB_URDF
```

Ejecuta el script principal:

```bash
python main.py
```

Si el archivo principal del proyecto se llama `load_icub.py`, ejecuta:

```bash
python load_icub.py
```

---

# Notas importantes para PyBullet

El cargador del robot debe usar el URDF indicado por `ICUB_URDF`.

Ejemplo mínimo:

```python
import os
import pybullet as p

ICUB_URDF = os.environ["ICUB_URDF"]

robot_id = p.loadURDF(
    ICUB_URDF,
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)
```

El uso de `p.URDF_USE_INERTIA_FROM_FILE` permite respetar las inercias definidas o parcheadas en el URDF.

---

# Problemas comunes

## `KeyError: 'ICUB_URDF'`

La variable de entorno no está definida.

Ubuntu:

```bash
export ICUB_URDF="/ruta/al/model_patched.urdf"
```

Windows:

```powershell
$env:ICUB_URDF="C:\ruta\al\model_patched.urdf"
```

## PyBullet no encuentra meshes

El URDF parcheado fue movido fuera de la carpeta original del modelo.
Mantén `model_patched.urdf` dentro de:

```text
iCubGazeboV2_5_visuomanip/
```

## El robot carga con advertencias de inercia

Ejecuta de nuevo el script de parcheo y asegúrate de que `ICUB_URDF` apunta a:

```text
model_patched.urdf
```

no a:

```text
model.urdf
```

## En Windows no cambia `ICUB_URDF` después de usar `setx`

`setx` solo afecta nuevas terminales.
Cierra y vuelve a abrir Miniforge Prompt.

---

# Flujo rápido

## Ubuntu

```bash
conda activate icub-pybullet
export ICUB_URDF="$CONDA_PREFIX/share/iCub/robots/iCubGazeboV2_5_visuomanip/model_patched.urdf"
python icub_simulator.py
```

## Windows

```powershell
conda activate icub-pybullet
$env:ICUB_URDF="C:\Users\TU_USUARIO\miniforge3\envs\icub-pybullet\Library\share\iCub\robots\iCubGazeboV2_5_visuomanip\model_patched.urdf"
python icub_simulator.py
```
