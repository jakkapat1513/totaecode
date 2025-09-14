"""
คำสั่งรันใน VS Code:
1. บันทึกไฟล์นี้เป็น auto_install_libs.py ในโฟลเดอร์โปรเจ็กต์
2. เปิด VS Code และเลือก Python interpreter (มุมล่างซ้าย, แนะนำ Python 3.8+)
3. เปิด terminal (Ctrl+`) และรัน: python auto_install_libs.py
   หรือกดปุ่ม Run ▶️ ใน VS Code
4. ตัวเลือกเพิ่มเติม:
   - ข้ามการติดตั้ง: python auto_install_libs.py --skip-install
   - จำลองการติดตั้ง: python auto_install_libs.py --dry-run
   - ระบุจำนวน workers: python auto_install_libs.py --max-workers 6

โค้ดนี้จะติดตั้งทุกไลบรารีที่จำเป็นและรันโค้ดโปรเจ็กต์หลักในไฟล์เดียว
"""

# นำเข้าโมดูลที่จำเป็น
import os
import sys
import subprocess
import importlib
import importlib.util
import logging
import json
from typing import Dict, Optional, Set, List, Tuple
import ast
import re
import argparse
import platform
import ctypes
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

# นำเข้าโมดูลเสริม (ถ้ามี)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from packaging import version as pkg_version
except ImportError:
    pkg_version = None

# ==================== การตั้งค่าเริ่มต้น ====================
# ตั้งค่า logging เพื่อบันทึก log ลงไฟล์และแสดงใน terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auto_installer.log")
    ]
)
logger = logging.getLogger(__name__)

# ==================== ฟังก์ชันตรวจสอบสิทธิ์ admin ====================
def is_admin() -> bool:
    """ตรวจสอบว่าสคริปต์รันด้วยสิทธิ์ admin หรือไม่"""
    try:
        if platform.system() == "Windows":
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0
    except AttributeError:
        return False

# ==================== ฟังก์ชันอัพเดท pip ====================
def self_update() -> bool:
    """อัพเดท pip เป็นเวอร์ชันล่าสุด"""
    try:
        logger.info("กำลังอัพเดท pip เป็นเวอร์ชันล่าสุด")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        if not is_admin():
            cmd.append("--user")
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_out, \
             tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_err:
            result = subprocess.run(cmd, stdout=tmp_out, stderr=tmp_err, text=True, check=True)
        logger.info("อัพเดท pip สำเร็จ")
        os.unlink(tmp_out.name)
        os.unlink(tmp_err.name)
        return True
    except subprocess.CalledProcessError as e:
        with open(tmp_err.name, 'r', encoding='utf-8') as f:
            error_output = f.read()
        logger.warning(f"ไม่สามารถอัพเดท pip: {error_output}")
        os.unlink(tmp_out.name)
        os.unlink(tmp_err.name)
        return False

# ==================== ฟังก์ชันจัดการการตั้งค่า ====================
def load_config(config_path: str = "install_config.json") -> Dict:
    """โหลดการตั้งค่าจากไฟล์ JSON ถ้ามี"""
    config = {
        "max_workers": 4,
        "ignore_patterns": [".venv", "__pycache__", ".git", "tests", "docs"],
        "dry_run": False,
        "cache_ttl": 3600,
        "network_timeout": 10,
        "retry_attempts": 3
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config.update(json.load(f))
            logger.info("โหลดไฟล์ตั้งค่าสำเร็จ")
        except Exception as e:
            logger.warning(f"ไม่สามารถโหลด {config_path}: {e}")
    return config

# ==================== ฟังก์ชันจัดการ PyPI Cache ====================
def load_pypi_cache(cache_file: str = "pypi_cache.json") -> Dict[str, str]:
    """โหลดแคชเวอร์ชัน PyPI จากไฟล์ JSON"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"ไม่สามารถโหลดแคช PyPI: {e}")
    return {}

def save_pypi_cache(cache: Dict[str, str], cache_file: str = "pypi_cache.json") -> None:
    """บันทึกแคชเวอร์ชัน PyPI ลงไฟล์ JSON"""
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        logger.debug(f"บันทึกแคช PyPI ไปที่ {cache_file}")
    except Exception as e:
        logger.warning(f"ไม่สามารถบันทึกแคช PyPI: {e}")

# ==================== ฟังก์ชันดึงเวอร์ชันล่าสุดจาก PyPI ====================
@lru_cache(maxsize=128)
def get_latest_version(pkg: str, timeout: int = 10, retry: int = 3) -> Optional[str]:
    """ดึงเวอร์ชันล่าสุดของแพ็กเกจจาก PyPI พร้อมแคชและ retry"""
    pypi_cache = load_pypi_cache()
    if pkg in pypi_cache:
        logger.debug(f"ใช้เวอร์ชันจากแคชสำหรับ {pkg}: {pypi_cache[pkg]}")
        return pypi_cache[pkg]

    for attempt in range(1, retry + 1):
        try:
            url = f"https://pypi.org/pypi/{pkg}/json"
            with urlopen(url, timeout=timeout) as response:
                if response.getcode() != 200:
                    raise HTTPError(url, response.getcode(), "HTTP Error", None, None)
                data = json.loads(response.read().decode('utf-8'))
                version = data["info"]["version"]
                pypi_cache[pkg] = version
                save_pypi_cache(pypi_cache)
                return version
        except (HTTPError, URLError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"พยายามครั้งที่ {attempt} ล้มเหลวในการดึงเวอร์ชันของ {pkg}: {e}")
    logger.warning(f"ไม่สามารถดึงเวอร์ชันของ {pkg} หลังจากลอง {retry} ครั้ง")
    return None

# ==================== ฟังก์ชันตรวจสอบความเข้ากันได้ ====================
def check_compatibility(pkg: str, version: Optional[str], installed_libs: Dict[str, str]) -> bool:
    """ตรวจสอบความเข้ากันได้ของแพ็กเกจกับไลบรารีที่ติดตั้งแล้ว"""
    if pkg in installed_libs and version and installed_libs[pkg] != version:
        logger.warning(f"เวอร์ชันไม่ตรงสำหรับ {pkg}: ติดตั้งแล้ว {installed_libs[pkg]}, ร้องขอ {version}")
        return False
    return True

# ==================== ฟังก์ชันติดตั้งแพ็กเกจ ====================
def install_package(pkg: str, version: Optional[str] = None, dry_run: bool = False, retry: int = 3) -> bool:
    """ติดตั้งแพ็กเกจ Python พร้อมตรวจสอบเวอร์ชันและ retry"""
    try:
        spec = importlib.util.find_spec(pkg)
        if spec:
            module = importlib.import_module(pkg)
            installed_version = getattr(module, '__version__', 'unknown')
            if version and installed_version == version:
                logger.info(f"{pkg} เวอร์ชัน {installed_version} ติดตั้งแล้ว")
                return True
            elif not version:
                logger.info(f"{pkg} (เวอร์ชัน {installed_version}) ติดตั้งแล้ว")
                return True
    except ImportError:
        pass

    if dry_run:
        logger.info(f"[Dry Run] จะติดตั้ง {pkg} {version or '(ล่าสุด)'}")
        return True

    pkg_spec = f"{pkg}=={version}" if version else pkg
    scopes = ["--user", ""] if pkg == "keyboard" else [""]
    for scope in scopes:
        for attempt in range(1, retry + 1):
            logger.info(f"ติดตั้ง {pkg_spec} {'ใน user scope' if scope else ''} (พยายามครั้งที่ {attempt}/{retry})")
            try:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_out, \
                     tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_err:
                    cmd = [sys.executable, "-m", "pip", "install"]
                    if scope:
                        cmd.append(scope)
                    cmd.append(pkg_spec)
                    result = subprocess.run(cmd, stdout=tmp_out, stderr=tmp_err, text=True, check=True)
                    logger.info(f"ติดตั้ง {pkg_spec} {'ใน user scope' if scope else ''} สำเร็จ")
                    os.unlink(tmp_out.name)
                    os.unlink(tmp_err.name)
                    return True
            except subprocess.CalledProcessError as e:
                with open(tmp_err.name, 'r', encoding='utf-8') as f:
                    error_output = f.read()
                logger.warning(f"พยายามครั้งที่ {attempt} ล้มเหลวในการติดตั้ง {pkg_spec} {'ใน user scope' if scope else ''}: {error_output}")
                os.unlink(tmp_out.name)
                os.unlink(tmp_err.name)
    logger.error(f"ไม่สามารถติดตั้ง {pkg_spec} หลังจากลอง {retry} ครั้งในทุก scope")
    return False

# ==================== ฟังก์ชันตรวจสอบและติดตั้งไลบรารี ====================
def ensure_library(lib: str, version: Optional[str] = None, dry_run: bool = False, retry: int = 3) -> bool:
    """ตรวจสอบและติดตั้งไลบรารีถ้ายังไม่ได้ติดตั้ง"""
    try:
        importlib.import_module(lib)
        logger.info(f"{lib} พร้อมใช้งานแล้ว")
        return True
    except ImportError:
        return install_package(lib, version, dry_run, retry)

# ==================== ฟังก์ชันสแกน imports จากไฟล์ Python ====================
def extract_imports(file_path: str) -> Set[str]:
    """ดึงรายชื่อไลบรารีที่ import จากไฟล์ Python"""
    libraries: Set[str] = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                source_code = f.read()
            except UnicodeDecodeError:
                logger.warning(f"ข้ามไฟล์ (encoding ไม่รองรับ): {file_path}")
                return set()

            tree = ast.parse(source_code, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        lib_name = name.name.split('.')[0]
                        libraries.add(lib_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        lib_name = node.module.split('.')[0]
                        libraries.add(lib_name)
        logger.debug(f"วิเคราะห์ {file_path}: พบ {len(libraries)} ไลบรารี")
        return libraries
    except Exception as e:
        logger.error(f"ไม่สามารถวิเคราะห์ {file_path}: {e}")
        return set()


# ==================== ฟังก์ชันสแกนโปรเจ็กต์ ====================
def scan_project_for_libraries(project_dir: str, ignore_patterns: List[str]) -> Set[str]:
    """สแกนไฟล์ Python ในโปรเจ็กต์เพื่อหาไลบรารีที่ใช้"""
    all_libraries: Set[str] = set()
    project_path = Path(project_dir)
    ignore_regex = [re.compile(pattern) for pattern in ignore_patterns]

    for file_path in project_path.rglob("*.py"):
        if str(file_path) == str(Path(sys.argv[0]).resolve()):  # ข้ามไฟล์ตัวเอง
            continue
        if any(any(regex.search(str(part)) for regex in ignore_regex) for part in file_path.parts):
            continue
        libraries = extract_imports(str(file_path))
        all_libraries.update(libraries)

    logger.info(f"พบ {len(all_libraries)} ไลบรารีในโปรเจ็กต์")
    return all_libraries

# ==================== ฟังก์ชันตรวจจับฮาร์ดแวร์ ====================
def detect_hardware() -> Tuple[str, Optional[str]]:
    """ตรวจจับประเภทฮาร์ดแวร์ (CPU, CUDA, MPS) และเวอร์ชัน CUDA ถ้ามี"""
    try:
        output = subprocess.check_output("nvidia-smi", shell=True, text=True)
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
        if match:
            cuda_version = match.group(1)
            short_version = "".join(cuda_version.split("."))
            logger.info(f"ตรวจพบ CUDA เวอร์ชัน: {cuda_version}")
            return "cuda", f"cu{short_version}"
    except Exception:
        pass
    try:
        if platform.system() == "darwin" and subprocess.run("sysctl -n machdep.cpu.brand_string", shell=True, capture_output=True, text=True).returncode == 0:
            logger.info("ตรวจพบ Apple Silicon (MPS)")
            return "mps", None
    except Exception:
        pass
    logger.info("ไม่พบ CUDA หรือ MPS ใช้ CPU")
    return "cpu", None

# ==================== ฟังก์ชันติดตั้ง PyTorch และ Torch-Geometric ====================
def install_torch_and_geometric(dry_run: bool = False, retry: int = 3) -> bool:
    """ติดตั้ง PyTorch และ Torch-Geometric ให้เข้ากับฮาร์ดแวร์"""
    cuda_mapping = {
        "10.2": "cu102", "11.0": "cu110", "11.1": "cu111", "11.2": "cu112",
        "11.3": "cu113", "11.4": "cu114", "11.5": "cu115", "11.6": "cu116",
        "11.7": "cu117", "11.8": "cu118", "12.0": "cu120", "12.1": "cu121",
        "12.2": "cu122", "12.3": "cu123", "12.4": "cu124", "12.5": "cu125",
        "12.6": "cu126", "12.7": "cu127", "12.8": "cu128"
    }
    try:
        import torch
        torch_version = torch.__version__.split("+")[0]
        logger.info(f"PyTorch {torch_version} ติดตั้งแล้ว")
        device_type, cuda_short = detect_hardware()
    except ImportError:
        if dry_run:
            logger.info("[Dry Run] จะติดตั้ง PyTorch")
            return True
        device_type, cuda_short = detect_hardware()
        index_url = None
        if device_type == "cuda" and cuda_short in cuda_mapping.values():
            index_url = f"https://download.pytorch.org/whl/{cuda_short}"
        elif device_type == "cuda":
            cuda_version = cuda_short[2:] if cuda_short else None
            if cuda_version:
                cuda_version_float = float(".".join(list(cuda_version)[:2]))
                supported_versions = sorted([float(v) for v in cuda_mapping.keys()])
                closest_version = min(supported_versions, key=lambda x: abs(x - cuda_version_float))
                cuda_short = cuda_mapping[str(closest_version)]
                index_url = f"https://download.pytorch.org/whl/{cuda_short}"
                logger.info(f"เลือก CUDA เวอร์ชันใกล้เคียงที่สุด: {closest_version}")
            else:
                cuda_short = "cpu"
        elif device_type == "mps":
            index_url = None
        else:
            cuda_short = "cpu"

        pkg_spec = "torch torchvision torchaudio"
        if index_url:
            pkg_spec += f" --index-url {index_url}"
        for attempt in range(1, retry + 1):
            logger.info(f"ติดตั้ง PyTorch (พยายามครั้งที่ {attempt}/{retry})")
            try:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_out, \
                     tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_err:
                    cmd = [sys.executable, "-m", "pip", "install"] + pkg_spec.split()
                    result = subprocess.run(cmd, stdout=tmp_out, stderr=tmp_err, text=True, check=True)
                    logger.info(f"ติดตั้ง PyTorch สำเร็จ")
                    os.unlink(tmp_out.name)
                    os.unlink(tmp_err.name)
                    import torch
                    torch_version = torch.__version__.split("+")[0]
                    break
            except subprocess.CalledProcessError as e:
                with open(tmp_err.name, 'r', encoding='utf-8') as f:
                    error_output = f.read()
                logger.warning(f"พยายามครั้งที่ {attempt} ล้มเหลวในการติดตั้ง PyTorch: {error_output}")
                os.unlink(tmp_out.name)
                os.unlink(tmp_err.name)
        else:
            logger.error(f"ไม่สามารถติดตั้ง PyTorch หลังจากลอง {retry} ครั้ง")
            return False

    logger.info(f"ตรวจพบอุปกรณ์: {device_type.upper()}")

    # ติดตั้ง Torch-Geometric และ dependencies
    required_versions = read_requirements()
    pyg_version = required_versions.get("torch_geometric") or get_latest_version("torch_geometric")
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_short or 'cpu' if device_type != 'mps' else 'cpu'}"
    pyg_libs = [
        "torch_scatter", "torch_sparse", "torch_cluster",
        "torch_spline_conv", "torch_geometric"
    ]
    for lib in pyg_libs:
        if dry_run:
            logger.info(f"[Dry Run] จะติดตั้ง {lib} {required_versions.get(lib) or '(ล่าสุด)'} สำหรับ {device_type}")
            continue
        version = required_versions.get(lib) or get_latest_version(lib)
        if pkg_version and version:
            latest_version = get_latest_version(lib)
            if latest_version and pkg_version.parse(version) < pkg_version.parse(latest_version):
                logger.warning(f"requirements.txt ระบุ {lib}=={version}, แต่ล่าสุดคือ {latest_version}. ติดตั้งเวอร์ชันล่าสุด")
                version = latest_version
        for attempt in range(1, retry + 1):
            try:
                ensure_library(lib, version, dry_run, retry=1)
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_out, \
                     tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_err:
                    cmd = [sys.executable, "-m", "pip", "install", "-f", pyg_url]
                    if version:
                        cmd.append(f"{lib}=={version}")
                    else:
                        cmd.append(lib)
                    result = subprocess.run(cmd, stdout=tmp_out, stderr=tmp_err, text=True, check=True)
                    logger.info(f"ติดตั้ง {lib} สำหรับ {device_type} สำเร็จ")
                    os.unlink(tmp_out.name)
                    os.unlink(tmp_err.name)
                    break
            except subprocess.CalledProcessError as e:
                with open(tmp_err.name, 'r', encoding='utf-8') as f:
                    error_output = f.read()
                logger.warning(f"พยายามครั้งที่ {attempt} ล้มเหลวในการติดตั้ง {lib}: {error_output}")
                os.unlink(tmp_out.name)
                os.unlink(tmp_err.name)
        else:
            logger.error(f"ไม่สามารถติดตั้ง {lib} หลังจากลอง {retry} ครั้ง")
            return False
    return True

# ==================== ฟังก์ชันอ่าน requirements.txt ====================
def read_requirements() -> Dict[str, str]:
    """อ่านความต้องการแพ็กเกจจาก requirements.txt"""
    requirements: Dict[str, str] = {}
    if os.path.exists("requirements.txt"):
        try:
            with open("requirements.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "==" in line and not line.startswith("#"):
                        pkg, ver = line.split("==")
                        requirements[pkg.strip()] = ver.strip()
            logger.info("อ่าน requirements.txt สำเร็จ")
        except Exception as e:
            logger.warning(f"ไม่สามารถอ่าน requirements.txt: {e}")
    return requirements

# ==================== ฟังก์ชันโค้ดโปรเจ็กต์หลัก ====================
def main_project_code():
    """โค้ดโปรเจ็กต์หลัก (แทนที่ด้วยโค้ดจริงของคุณ)"""
    logger.info("เริ่มรันโค้ดโปรเจ็กต์หลัก")
    try:
        # ตัวอย่างการใช้ไลบรารีที่ติดตั้ง
        import gym
        import pandas as pd
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        # ตัวอย่างโค้ด
        logger.info("ทดสอบการใช้ไลบรารี")
        env = gym.make("CartPole-v1")
        observation = env.reset()
        logger.info(f"Gym environment: {observation}")

        df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        logger.info(f"Pandas DataFrame:\n{df}")

        array = np.array([1, 2, 3])
        logger.info(f"NumPy array: {array}")

        tensor = torch.tensor([1, 2, 3])
        logger.info(f"PyTorch tensor: {tensor}")

        model = LinearRegression()
        model.fit(df[["x"]], df["y"])
        logger.info(f"Scikit-learn model coefficient: {model.coef_}")

        plt.plot(df["x"], df["y"], 'o')
        plt.title("ตัวอย่างกราฟ")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("example_plot.png")
        plt.close()
        logger.info("บันทึกกราฟตัวอย่างสำเร็จ")

        # คุณสามารถแทนที่โค้ดนี้ด้วยโค้ดโปรเจ็กต์จริงของคุณ
        print("โค้ดโปรเจ็กต์หลักทำงานสำเร็จ")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในโค้ดโปรเจ็กต์หลัก: {e}")
        raise

# ==================== ฟังก์ชันติดตั้งทุกไลบรารีและรันโปรเจ็กต์ ====================
def auto_install_all(project_dir: str = ".", dry_run: bool = False, max_workers: int = 4, skip_install: bool = False):
    ensure_library('packaging')
    from packaging import version as pkg_version

    pip = None  # กำหนดตัวแปร pip ก่อนใช้งาน

    # กำหนด ignore_patterns เป็น list ว่าง (หรือเพิ่ม pattern ที่ต้องการกรองได้)
    ignore_patterns = []

    if skip_install and not dry_run:
        logger.info("ข้ามการติดตั้งไลบรารี ไปรันโค้ดโปรเจ็กต์หลัก")
        main_project_code()
        return

    required_versions = read_requirements()
    project_libs = scan_project_for_libraries(project_dir, ignore_patterns)
    base_libs = {
        "numpy", "pandas", "scikit-learn", "joblib", "tqdm", "matplotlib",
        "seaborn", "plotly", "ccxt", "binance", "websockets", "yfinance",
        "requests", "psutil", "ta", "pandas_ta", "gym", "openpyxl",
        "keyboard", "bayesian-optimization", "darts", "packaging"
    }
    standard_libs = {
        "sys", "os", "logging", "typing", "subprocess", "importlib", "ast",
        "json", "argparse", "urllib", "re", "functools", "concurrent", "pathlib",
        "platform", "ctypes", "datetime", "asyncio", "sqlite3", "tempfile"
    }

    all_libs = base_libs.union(project_libs) - standard_libs

    installed_libs: Dict[str, str] = {}

    # อัปเกรด pip อย่างปลอดภัย
    try:
        import subprocess
        subprocess.check_call(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])
        pip = "upgraded"
    except Exception as e:
        logger.warning(f"ไม่สามารถอัปเกรด pip ได้: {e}")

    # ใส่โค้ดติดตั้งไลบรารีเพิ่มเติมได้ที่นี่
    # ...


    all_libs = base_libs.union(project_libs) - standard_libs

    installed_libs: Dict[str, str] = {}

    # ตัวอย่างการอัปเดต pip ใน virtualenv แบบปลอดภัย
    try:
        import subprocess
        subprocess.check_call(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])
        pip = "upgraded"
    except Exception as e:
        logger.warning(f"ไม่สามารถอัปเกรด pip ได้: {e}")

    # ... โค้ดติดตั้งไลบรารีอื่น ๆ ตามที่จำเป็น


    required_versions = read_requirements()
    project_libs = scan_project_for_libraries(project_dir, ignore_patterns)
    base_libs = {
        "numpy", "pandas", "scikit-learn", "joblib", "tqdm", "matplotlib",
        "seaborn", "plotly", "ccxt", "binance", "websockets", "yfinance",
        "requests", "psutil", "ta", "pandas_ta", "gym", "openpyxl",
        "keyboard", "bayesian-optimization", "darts", "packaging"
    }
    standard_libs = {
        "sys", "os", "logging", "typing", "subprocess", "importlib", "ast",
        "json", "argparse", "urllib", "re", "functools", "concurrent", "pathlib",
        "platform", "ctypes", "datetime", "asyncio", "sqlite3", "tempfile"
    }
    all_libs = base_libs.union(project_libs) - standard_libs

    installed_libs: Dict[str, str] = {}
    failed_libs: List[str] = []

    def install_with_progress(lib: str) -> bool:
        """ติดตั้งไลบรารีพร้อมเปรียบเทียบเวอร์ชัน"""
        latest_version = get_latest_version(lib, timeout=timeout, retry=retry)
        req_version = required_versions.get(lib)
        if req_version and latest_version and pkg_version:
            if pkg_version.parse(req_version) < pkg_version.parse(latest_version):
                logger.warning(f"requirements.txt ระบุ {lib}=={req_version}, แต่ล่าสุดคือ {latest_version}. ติดตั้งเวอร์ชันล่าสุด")
                version = latest_version
            else:
                version = req_version
        else:
            version = latest_version or req_version
        if not version:
            logger.error(f"ไม่สามารถหาเวอร์ชันของ {lib}")
            return False
        if check_compatibility(lib, version, installed_libs):
            success = ensure_library(lib, version, dry_run, retry)
            if success and not dry_run:
                installed_libs[lib] = version
            return success
        failed_libs.append(lib)
        return False

    logger.info(f"เริ่มติดตั้ง {len(all_libs)} ไลบรารี (Dry Run: {dry_run})")
    if not dry_run:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(install_with_progress, lib): lib for lib in all_libs}
            progress = tqdm(futures, total=len(futures), desc="กำลังติดตั้ง", disable=not tqdm)
            for future in as_completed(futures):
                lib = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"เกิดข้อผิดพลาดในการติดตั้ง {lib}: {e}")
                    failed_libs.append(lib)
                if progress:
                    progress.update(1)
            if progress:
                progress.close()

    if "torch" in all_libs or "torch_geometric" in all_libs:
        if not install_torch_and_geometric(dry_run, retry):
            failed_libs.append("torch/torch_geometric")

    if failed_libs:
        logger.warning(f"ไม่สามารถติดตั้ง: {', '.join(failed_libs)}")
    else:
        logger.info("ติดตั้งไลบรารีทั้งหมดสำเร็จ")

    # รันโค้ดโปรเจ็กต์หลัก
    if not dry_run:
        try:
            main_project_code()
            logger.info("รันโค้ดโปรเจ็กต์หลักสำเร็จ")
        except Exception as e:
            logger.error(f"รันโค้ดโปรเจ็กต์หลักล้มเหลว: {e}")

# ==================== ฟังก์ชันหลัก ====================
def main():
    """จัดการ command-line arguments และเริ่มการติดตั้งพร้อมรันโปรเจ็กต์"""
    parser = argparse.ArgumentParser(description="ติดตั้งไลบรารี Python และรันโค้ดโปรเจ็กต์โดยอัตโนมัติ")
    parser.add_argument("--project-dir", default=".", help="โฟลเดอร์โปรเจ็กต์ที่ต้องการสแกน")
    parser.add_argument("--dry-run", action="store_true", help="จำลองการติดตั้งโดยไม่ดำเนินการจริง")
    parser.add_argument("--max-workers", type=int, default=4, help="จำนวน workers สูงสุดสำหรับการติดตั้งแบบขนาน")
    parser.add_argument("--skip-install", action="store_true", help="ข้ามการติดตั้งไลบรารีและรันโค้ดโปรเจ็กต์ทันที")
    args = parser.parse_args()

    try:
        auto_install_all(args.project_dir, args.dry_run, args.max_workers, args.skip_install)
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในโปรแกรม: {e}")
        sys.exit(1)

# ===============================================
# config.py
# ไฟล์ตั้งค่า CONFIG กลาง ใช้ร่วมกันทุกคลาส
# สามารถดึงค่าหรือแก้ไขค่าได้แบบเรียลไทม์
# ===============================================

import logging

logging.basicConfig(level=logging.INFO)

class GlobalConfig:
    CONFIG = {
        # ===============================
        # API and Exchange Parameters
        # พารามิเตอร์สำหรับ API และการเชื่อมต่อ Binance
        # ===============================
        'binance_api_key': 'NcciO0GXlpxbpsJkRFRuRF4IiW03Ot0gIj92vZQWAtWlvc1BEtYqJezhyS6E70us',            # คีย์ API สำหรับเชื่อมต่อ Binance Futures (ปรับให้เป็นคีย์จริงก่อนใช้งาน)
        'binance_api_secret': 'c3e7tt5cnL4CNZ1dearAJESPWQTZD4J1RdxLTSkdDidwKitEf3nEVip8EMhIxViv',      # รหัสลับ API สำหรับ Binance Futures (ปรับให้เป็นรหัสลับจริงก่อนใช้งาน)
        'dry_run': True,                                 # True = จำลองการเรียก API และเทรด (ไม่ส่งจริง), False = เรียก API และเทรดจริง
        'max_api_retries': 10,                             # จำนวนครั้งสูงสุดที่พยายามเรียก API ใหม่เมื่อล้มเหลว
        'api_timeout': 30,                                 # วินาทีที่รอการตอบกลับจาก API ก่อน timeout
        'rate_limit_per_minute': 2400,                     # จำนวนคำขอ API ต่อนาที (ตาม Binance Futures ในปี 2025: 2400/min per IP)
        'margin_mode': 'isolated',                            # โหมด margin ('cross' หรือ 'isolated')
        'min_leverage': 5,                                 # Leverage ขั้นต่ำที่ยอมรับ (ปรับอัตโนมัติแต่ไม่ต่ำกว่านี้)
        'trailing_callback_rate': 0.5,                     # เปอร์เซ็นต์ callback สำหรับ Trailing Stop และ Take Profit (เช่น 0.5%)
        'trailing_update_interval': 60,                    # ระยะเวลา (วินาที) สำหรับการอัพเดท Trailing SL/TP อัตโนมัติ
        'sync_time_interval': 3600,                        # ระยะเวลา (วินาที) สำหรับการซิงค์เวลากับเซิร์ฟเวอร์ Binance
        'futures_weight': 0.9,                             # น้ำหนักการใช้งาน API สำหรับ Futures (เพื่อ allocate rate limit)

        # ===============================
        # WebSocket Parameters
        # พารามิเตอร์สำหรับ WebSocket และการจัดการข้อมูลเรียลไทม์
        # ===============================
        'ws_url': 'wss://fstream.binance.com/ws',          # URL หลักสำหรับ WebSocket Market Streams ของ Binance Futures
        'ws_backup_url': 'wss://stream.binance.com:9443/ws',  # URL สำรองสำหรับ WebSocket
        'max_reconnects': 10,                              # จำนวนครั้งสูงสุดในการ reconnect ก่อนใช้ข้อมูลสำรอง
        'reconnect_delay_max': 60,                         # วินาทีสูงสุดในการรอ reconnect
        'ws_timeout': 10,                                  # วินาที timeout สำหรับการรับข้อความจาก WebSocket
        'db_path': 'ws_backup.db',                         # Path ของฐานข้อมูล SQLite สำหรับ备份ข้อมูล WebSocket
        'cache_size_max': 1000,                            # ขนาดสูงสุดของ cache สำหรับข้อมูลล่าสุด
        'data_retention_limit': 100,                       # จำนวนแถวสูงสุดที่เก็บใน SQLite ต่อ symbol (สำหรับข้อมูลล่าสุด)
        'multi_tf_list': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'],  # รายการ timeframe สำหรับ aggregation
        'historical_years': 5,                             # จำนวนปีของข้อมูลย้อนหลังสำหรับโหมดจำลอง

        # ===============================
        # Model Specific Parameters
        # พารามิเตอร์เฉพาะสำหรับโมเดล AI (เช่น SSD, EvoGAN)
        # ===============================
        'min_volatility_threshold': 0.005,                 # ความผันผวนขั้นต่ำที่ยอมรับสำหรับปรับ learning rate (ใช้ใน SSD)
        'nas_iterations': 100,                             # จำนวน iterations สำหรับ Neural Architecture Search (ใช้ใน EvoGAN)

        # ===============================
        # Trading and Risk Management Parameters
        # พารามิเตอร์การเทรดและการบริหารความเสี่ยง
        # ===============================
        'profit_lock_percentage': 0.05,                    # เปอร์เซ็นต์กำไรที่ล็อกเมื่อถึงเป้า (เช่น ล็อกกำไรที่ 5%)
        'loss_strategy': 'dynamic',                        # กลยุทธ์ตัดขาดทุน ('dynamic' = ปรับตาม ATR เพื่อความยืดหยุ่นตามตลาด)
        'stop_loss_percentage': 0.005,                     # เปอร์เซ็นต์ขาดทุนเริ่มต้น (เช่น 0.5% สำหรับ stop loss เริ่มต้น, ปรับตามกลยุทธ์)
        'cut_loss_threshold': 0.2,                         # ขีดจำกัดขาดทุนสูงสุด (เป็น fraction เช่น 20% ของทุน)
        'risk_per_trade': 0.2,                             # ความเสี่ยงต่อการเทรดแต่ละครั้ง (เช่น 20% ของทุน, ปรับตาม KPI เพื่อควบคุม drawdown)
        'max_drawdown': 0.2,                               # เปอร์เซ็นต์ drawdown สูงสุดที่ยอมรับได้ (เช่น 20% ของทุนรวม)
        'liquidity_threshold': 1000000,                    # ปริมาณการซื้อขายขั้นต่ำ (หน่วย USDT เพื่อให้แน่ใจว่ามี liquidity เพียงพอ)

        # ===============================
        # Financial Management Parameters
        # พารามิเตอร์ด้านการจัดการการเงิน
        # ===============================
        'initial_balance': 100,                            # ยอดเงินเริ่มต้นในบัญชี (หน่วย USDT, ใช้สำหรับการคำนวณ reward และ reinvest)
        'reinvest_profits': True,                          # True = นำกำไรมา reinvest เพื่อทบต้นทุน, False = ไม่ reinvest (เก็บกำไรแยก)

        # ===============================
        # Simulation Mode Parameters
        # พารามิเตอร์สำหรับโหมดจำลอง (dry_run)
        # ===============================
        'sim_volatility': 0.02,                            # ความผันผวนในโหมดจำลอง (เช่น 2% สำหรับการสุ่มราคา)
        'sim_trend': 0.001,                                # แนวโน้มราคาในโหมดจำลอง (เช่น 0.1% ต่อ step สำหรับการจำลอง uptrend/downtrend)
        'sim_spike_chance': 0.05,                          # โอกาสเกิดการพุ่งของราคาในโหมดจำลอง (เช่น 5% โอกาส spike)

        # ===============================
        # Model Training and Optimization Parameters
        # พารามิเตอร์ฝึกโมเดล AI และการปรับแต่ง
        # ===============================
        'auto_ml_interval': 500,                           # จำนวน steps ก่อนฝึก ML อัตโนมัติ (เช่น ทุก 500 steps เพื่อปรับ hyperparameter)
        'rl_train_interval': 200,                          # จำนวน steps ก่อนฝึก Reinforcement Learning (เช่น ทุก 200 steps เพื่ออัพเดท policy)
        'checkpoint_interval': 360,                        # จำนวน steps ก่อนบันทึก checkpoint (เช่น ทุก 360 steps หรือ 6 ชั่วโมง)
        'bayes_opt_steps': 10,                             # จำนวน iterations สำหรับ Bayesian Optimization (เพื่อหา hyperparameter ที่ดีที่สุด)
        'gnn_update_interval': 900,                        # วินาทีที่อัพเดท GNN graph (เช่น ทุก 15 นาที เพื่อวิเคราะห์ความสัมพันธ์เหรียญ)
        'madrl_agent_count': 50,                           # จำนวน agent สูงสุดใน MADRL (ตามจำนวนเหรียญที่เทรด เพื่อจัดการ multi-agent)
        'max_coins_per_trade': 15,                         # จำนวนเหรียญสูงสุดที่เทรดพร้อมกัน (เพื่อจำกัด portfolio size)
        'min_coins_per_trade': 6,                          # จำนวนเหรียญขั้นต่ำที่เทรดพร้อมกัน (เพื่อกระจายความเสี่ยง)
        'maml_lr_inner': 0.01,                             # Learning rate ภายในสำหรับ MAML (เพื่อ fine-tuning รวดเร็ว)
        'maml_lr_outer': 0.001,                            # Learning rate ภายนอกสำหรับ MAML (เพื่ออัพเดท meta-model)
        'maml_steps': 5,                                   # จำนวน steps สำหรับ fine-tuning ใน MAML (เพื่อการเรียนรู้แบบ few-shot)

        # ===============================
        # KPI and Performance Parameters
        # พารามิเตอร์เกี่ยวกับ KPI และประสิทธิภาพ
        # ===============================
        'target_kpi_daily': 100000.0,                      # เป้าหมายกำไรรายวัน (หน่วย USDT, เพื่อวัดประสิทธิภาพระบบ)
        'min_daily_kpi': 50000.0,                          # เป้าหมายกำไรขั้นต่ำระหว่างวัน (50% ของ target เพื่อ trigger การปรับกลยุทธ์)

        # ===============================
        # Resource and Bug Management Parameters
        # พารามิเตอร์การจัดการทรัพยากรและบั๊ก
        # ===============================
        'auto_bug_fix': True,                              # True = แก้บั๊กอัตโนมัติเมื่อเกิดข้อผิดพลาด (เช่น CUDA OOM หรือ rate limit)
        'bug_fix_attempts': 5,                             # จำนวนครั้งสูงสุดที่พยายามแก้บั๊ก (ก่อนหยุดระบบ)
        'resource_adaptive': True,                         # True = ปรับการใช้ทรัพยากร (CPU/RAM) อัตโนมัติ (เพื่อป้องกัน overload)
        'min_ram_reserve_mb': 1024,                        # จำนวน MB ของ RAM ที่สำรองไว้ (เพื่อป้องกัน memory error)
        'min_cpu_idle_percent': 20,                        # เปอร์เซ็นต์ CPU ว่างขั้นต่ำ (เพื่อป้องกัน CPU overload)

        # ===============================
        # System Status and Logging
        # พารามิเตอร์เกี่ยวกับสถานะระบบและการบันทึก
        # ===============================
        'system_running': False,                           # สถานะการรันระบบ (True = ทำงาน, False = หยุด)
        'trade_log_file': 'trade_log.xlsx',                # ไฟล์บันทึกการเทรด (สำหรับบันทึก transaction history)
        'log_level': 'INFO',                               # ระดับการบันทึก log ('DEBUG', 'INFO', 'WARNING', 'ERROR') สำหรับระบบโดยรวม

        # ===============================
        # Additional Parameters for MADRL
        # พารามิเตอร์เพิ่มเติมสำหรับ MADRL
        # ===============================
        'max_leverage_per_symbol': {'BTCUSDT': 125, 'ETHUSDT': 125},  # Leverage สูงสุดต่อ symbol (default 125)
    }

    @classmethod
    def get(cls, key: str, default=None):
        """
        ดึงค่าจาก CONFIG
        :param key: str → ชื่อคีย์ที่ต้องการดึงค่า
        :param default: ค่าเริ่มต้น ถ้าไม่พบ key
        """
        return cls.CONFIG.get(key, default)

    @classmethod
    def set(cls, key: str, value):
        """
        เปลี่ยนค่าของ CONFIG แบบเรียลไทม์
        :param key: str → ชื่อคีย์ที่ต้องการแก้ไข
        :param value: ค่าใหม่ที่ต้องการตั้ง
        """
        cls.CONFIG[key] = value
        logging.info(f"อัพเดทค่า CONFIG: {key} เป็น {value}")

CONFIG = GlobalConfig.CONFIG
# ===============================================
# APIManager Class
# จัดการการเชื่อมต่อ API กับ Binance Futures
# ใช้ GlobalConfig สำหรับค่าพารามิเตอร์ทั้งหมด
# ===============================================

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งใน terminal ก่อนใช้งาน)
# pip install ccxt tenacity

import logging
import logging.handlers
import time
import asyncio
from collections import deque
from typing import Optional, Dict, Any
import ccxt.async_support as ccxt_async
from tenacity import retry, wait_exponential, stop_after_attempt
#from config import GlobalConfig

# การตั้งค่าระบบบันทึก log เฉพาะสำหรับคลาสนี้
log_level_str = GlobalConfig.get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('apimanager.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

class APIManager:
    """จัดการการเชื่อมต่อและคำสั่งผ่าน Binance Futures API"""

    def __init__(self):
        self.weight_used: int = 0
        self.weight_limit: int = GlobalConfig.get('rate_limit_per_minute')
        self.last_reset: float = time.time()
        self.is_rate_limited: bool = False
        self.ban_until: float = 0
        self.request_count: int = 0
        self.rate_limit_status: Dict[str, int] = {}
        self.kpi_priority_weight: float = 0
        self.api_call_timestamps: deque = deque(maxlen=GlobalConfig.get('rate_limit_per_minute'))
        self.time_offset: int = 0
        self.last_time_sync: float = time.time()
        self.max_leverage_per_symbol: Dict[str, int] = GlobalConfig.get('max_leverage_per_symbol', {})
        self.trailing_orders: Dict[str, Dict[str, Any]] = {}
        self.markets: Optional[Dict] = None
        self.exchange = ccxt_async.binance({
            'apiKey': GlobalConfig.get('binance_api_key'),
            'secret': GlobalConfig.get('binance_api_secret'),
            'enableRateLimit': True,
            'timeout': GlobalConfig.get('api_timeout') * 1000,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            },
            'rateLimit': int(60000 / GlobalConfig.get('rate_limit_per_minute'))
        })

    async def close(self):
        """ปิดการเชื่อมต่อ exchange"""
        if self.exchange:
            await self.exchange.close()

    async def sync_time_with_exchange(self) -> bool:
        """ซิงโครไนซ์เวลากับเซิร์ฟเวอร์ Binance"""
        if GlobalConfig.get('dry_run'):
            logging.info("จำลองการซิงค์เวลา: timeOffset = 0")
            self.time_offset = 0
            self.last_time_sync = time.time()
            return True
        await self.rate_limit_control()
        try:
            server_time = await self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            self.last_time_sync = time.time()
            logging.info(f"ซิงค์เวลาสำเร็จ: timeOffset = {self.time_offset} มิลลิวินาที")
            return True
        except Exception as e:
            logging.error(f"ซิงค์เวลาล้มเหลว: {e}")
            self.time_offset = 0
            return False

    async def get_adjusted_timestamp(self) -> int:
        """คืนค่า timestamp ที่ปรับตาม timeOffset (มิลลิวินาที)"""
        return int(time.time() * 1000) + self.time_offset

    async def update_weight(self, response: Any):
        """อัพเดทน้ำหนัก API และรีเซ็ตทุก 60 วินาที"""
        if time.time() - self.last_reset >= 60:
            self.weight_used = 0
            self.rate_limit_status = {}
            self.last_reset = time.time()

    async def rate_limit_control(self):
        """ควบคุม rate limit เพื่อป้องกันการถูกแบนจาก Binance"""
        now = time.time()
        if now - self.last_time_sync >= GlobalConfig.get('sync_time_interval'):
            await self.sync_time_with_exchange()
        if self.is_rate_limited and now < self.ban_until:
            wait_time = self.ban_until - now
            logging.warning(f"ถูกจำกัด IP รอ {wait_time:.2f} วินาที")
            await asyncio.sleep(wait_time)
            self.is_rate_limited = False
        if self.weight_used >= self.weight_limit * 0.9:
            wait_time = 60 - (now - self.last_reset)
            if wait_time > 0:
                logging.warning(f"น้ำหนัก API ใกล้เต็ม รอ {wait_time:.2f} วินาที")
                await asyncio.sleep(wait_time)
            self.weight_used = 0
            self.last_reset = now
        self.api_call_timestamps.append(now)
        self.request_count += 1
        if len(self.api_call_timestamps) >= self.weight_limit:
            wait_time = (60 / self.weight_limit) - (now - self.api_call_timestamps[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def optimize_api_usage(self, kpi_tracker: Optional[Any] = None):
        """ปรับการใช้งาน API ตาม KPI และน้ำหนักที่ใช้"""
        if self.weight_used > self.weight_limit * 0.5:
            logging.info("ปรับ interval การเรียก API ช้าลงเนื่องจากน้ำหนักใช้งานเกินครึ่ง")
        if kpi_tracker and hasattr(kpi_tracker, 'total_profit') and kpi_tracker.total_profit < GlobalConfig.get('min_daily_kpi'):
            self.kpi_priority_weight += 0.1
            logging.info(f"เพิ่มน้ำหนัก priority สำหรับ KPI: {self.kpi_priority_weight:.2f}")
        self.request_count = 0

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(GlobalConfig.get('max_api_retries')))
    async def fetch_max_leverage(self) -> Dict[str, int]:
        """ดึงข้อมูล leverage สูงสุดสำหรับแต่ละเหรียญ"""
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info("จำลองการดึง max_leverage")
            return self.max_leverage_per_symbol
        try:
            info = await self.exchange.fetch_exchange_info(params={'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()})
            max_leverage = {}
            for symbol_info in info['symbols']:
                symbol = symbol_info['symbol']
                if symbol.endswith('USDT'):
                    max_lev = 125
                    for filt in symbol_info.get('filters', []):
                        if filt['filterType'] == 'LEVERAGE_BRACKET':
                            brackets = filt.get('brackets', [])
                            if brackets:
                                max_lev = max(br.get('initialLeverage', 125) for br in brackets)
                            break
                    max_leverage[symbol] = int(max_lev)
            self.max_leverage_per_symbol = max_leverage
            GlobalConfig.set('max_leverage_per_symbol', max_leverage)
            await self.update_weight(info)
            logging.info(f"ดึง max_leverage สำเร็จ: {len(max_leverage)} เหรียญ")
            return max_leverage
        except Exception as e:
            logging.error(f"ดึง max_leverage ล้มเหลว: {e}")
            raise

    async def set_margin_mode(self, symbol: str) -> Optional[Dict]:
        """ตั้งค่า margin mode (CROSS หรือ ISOLATED) สำหรับเหรียญ"""
        mode = GlobalConfig.get('margin_mode').upper() if GlobalConfig.get('margin_mode').upper() in ['CROSS', 'ISOLATED'] else 'CROSS'
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการตั้ง margin mode สำหรับ {symbol} เป็น {mode}")
            return {'status': 'success'}
        try:
            response = await self.exchange.set_margin_mode(mode.lower(), symbol, params={'type': 'future', 'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()})
            await self.update_weight(response)
            logging.info(f"ตั้ง margin mode สำหรับ {symbol} เป็น {mode} สำเร็จ")
            return response
        except Exception as e:
            logging.error(f"ตั้ง margin mode สำหรับ {symbol} ล้มเหลว: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> Optional[Dict]:
        """ตั้งค่า leverage โดยจำกัดระหว่าง min_leverage และ max_leverage"""
        max_lev = self.max_leverage_per_symbol.get(symbol, 125)
        adjusted_leverage = max(GlobalConfig.get('min_leverage'), min(int(leverage), max_lev))
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการตั้ง leverage สำหรับ {symbol} เป็น {adjusted_leverage}")
            return {'status': 'success'}
        try:
            response = await self.exchange.set_leverage(adjusted_leverage, symbol, params={'type': 'future', 'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()})
            await self.update_weight(response)
            logging.info(f"ตั้ง leverage สำหรับ {symbol} เป็น {adjusted_leverage} สำเร็จ")
            return response
        except Exception as e:
            logging.error(f"ตั้ง leverage สำหรับ {symbol} ล้มเหลว: {e}")
            return None

    async def create_limit_order_with_trailing(self, symbol: str, side: str, amount: float, price: float, callback_rate: Optional[float] = None, activation_price: Optional[float] = None) -> Optional[Dict]:
        """สร้าง limit order พร้อม Trailing Stop Loss และ Take Profit"""
        callback_rate = callback_rate or GlobalConfig.get('trailing_callback_rate')
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการสร้าง limit order พร้อม trailing SL/TP สำหรับ {symbol}: {side} {amount} @{price}, callback {callback_rate}%")
            return {'order_id': 'simulated_order_id', 'sl_order_id': 'simulated_sl_id', 'tp_order_id': 'simulated_tp_id', 'status': 'success'}
        try:
            order_params = {
                'timeInForce': 'GTC',
                'recvWindow': 5000,
                'timestamp': await self.get_adjusted_timestamp()
            }
            order = await self.exchange.create_order(symbol, 'limit', side.lower(), amount, price, order_params)
            await self.update_weight(order)
            logging.info(f"สร้าง limit order สำหรับ {symbol} สำเร็จ: ID {order['id']}")
            opposite_side = 'sell' if side.lower() == 'buy' else 'buy'
            sl_params = {
                'type': 'TRAILING_STOP_MARKET',
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'recvWindow': 5000,
                'timestamp': await self.get_adjusted_timestamp()
            }
            if activation_price:
                sl_params['activationPrice'] = activation_price if side.lower() == 'buy' else price * (1 + callback_rate / 100)
            sl_order = await self.exchange.create_order(symbol, 'market', opposite_side, amount, None, sl_params)
            await self.update_weight(sl_order)
            tp_params = {
                'type': 'TRAILING_STOP_MARKET',
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'recvWindow': 5000,
                'timestamp': await self.get_adjusted_timestamp()
            }
            if activation_price:
                tp_params['activationPrice'] = activation_price if side.lower() == 'sell' else price * (1 - callback_rate / 100)
            tp_order = await self.exchange.create_order(symbol, 'market', opposite_side, amount, None, tp_params)
            await self.update_weight(tp_order)
            self.trailing_orders[symbol] = {
                'order_id': order['id'],
                'sl_order_id': sl_order['id'],
                'tp_order_id': tp_order['id'],
                'last_price': price,
                'side': side.lower(),
                'amount': amount
            }
            logging.info(f"สร้าง trailing SL/TP สำเร็จ: SL ID {sl_order['id']}, TP ID {tp_order['id']}")
            return {
                'order_id': order['id'],
                'sl_order_id': sl_order['id'],
                'tp_order_id': tp_order['id'],
                'status': 'success'
            }
        except Exception as e:
            logging.error(f"สร้าง limit order พร้อม trailing SL/TP สำหรับ {symbol} ล้มเหลว: {e}")
            return None

    async def update_trailing_orders(self, symbol: str, current_price: float, volatility: Optional[float] = None) -> bool:
        """อัพเดท Trailing Stop Loss และ Take Profit ตามราคาปัจจุบันและความผันผวน"""
        if symbol not in self.trailing_orders:
            return False
        callback_rate = GlobalConfig.get('trailing_callback_rate')
        if volatility:
            vol_threshold = GlobalConfig.get('min_volatility_threshold', 0.005)
            callback_rate = max(callback_rate, min(2.0, callback_rate * (1 + volatility / vol_threshold)))
        order_info = self.trailing_orders[symbol]
        side = order_info['side']
        amount = order_info['amount']
        opposite_side = 'sell' if side == 'buy' else 'buy'
        try:
            await self.cancel_order(symbol, order_info['sl_order_id'])
            await self.cancel_order(symbol, order_info['tp_order_id'])
            sl_params = {
                'type': 'TRAILING_STOP_MARKET',
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'recvWindow': 5000,
                'timestamp': await self.get_adjusted_timestamp()
            }
            sl_order = await self.exchange.create_order(symbol, 'market', opposite_side, amount, None, sl_params)
            await self.update_weight(sl_order)
            tp_params = {
                'type': 'TRAILING_STOP_MARKET',
                'callbackRate': callback_rate,
                'reduceOnly': True,
                'recvWindow': 5000,
                'timestamp': await self.get_adjusted_timestamp()
            }
            tp_order = await self.exchange.create_order(symbol, 'market', opposite_side, amount, None, tp_params)
            await self.update_weight(tp_order)
            self.trailing_orders[symbol].update({
                'sl_order_id': sl_order['id'],
                'tp_order_id': tp_order['id'],
                'last_price': current_price
            })
            logging.info(f"อัพเดท trailing SL/TP สำหรับ {symbol} สำเร็จ: SL ID {sl_order['id']}, TP ID {tp_order['id']}")
            return True
        except Exception as e:
            logging.error(f"อัพเดท trailing SL/TP สำหรับ {symbol} ล้มเหลว: {e}")
            return False

    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: Dict = {}) -> Optional[Dict]:
        """สร้าง limit order ธรรมดา"""
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการสร้าง limit order สำหรับ {symbol}: {side} {amount} @{price}")
            return {'id': 'simulated_order_id', 'status': 'success'}
        try:
            full_params = {**params, 'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()}
            order = await self.exchange.create_order(symbol, 'limit', side.lower(), amount, price, full_params)
            await self.update_weight(order)
            logging.info(f"สร้าง limit order สำหรับ {symbol} สำเร็จ: ID {order['id']}")
            return order
        except Exception as e:
            logging.error(f"สร้าง limit order สำหรับ {symbol} ล้มเหลว: {e}")
            return None

    async def create_stop_order(self, symbol: str, side: str, amount: float, stop_price: float, params: Dict = {'reduceOnly': True}) -> Optional[Dict]:
        """สร้าง stop order (Stop Loss)"""
        full_params = {
            **params,
            'type': 'STOP_MARKET',
            'stopPrice': stop_price,
            'recvWindow': 5000,
            'timestamp': await self.get_adjusted_timestamp()
        }
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการสร้าง stop order สำหรับ {symbol}: {side} @{stop_price}")
            return {'id': 'simulated_stop_id', 'status': 'success'}
        try:
            order = await self.exchange.create_order(symbol, 'market', side.lower(), amount, None, full_params)
            await self.update_weight(order)
            logging.info(f"สร้าง stop order สำหรับ {symbol} สำเร็จ: ID {order['id']}")
            return order
        except Exception as e:
            logging.error(f"สร้าง stop order สำหรับ {symbol} ล้มเหลว: {e}")
            return None

    async def create_take_profit_order(self, symbol: str, side: str, amount: float, tp_price: float, params: Dict = {'reduceOnly': True}) -> Optional[Dict]:
        """สร้าง take profit order"""
        full_params = {
            **params,
            'type': 'TAKE_PROFIT_MARKET',
            'stopPrice': tp_price,
            'recvWindow': 5000,
            'timestamp': await self.get_adjusted_timestamp()
        }
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการสร้าง take profit order สำหรับ {symbol}: {side} @{tp_price}")
            return {'id': 'simulated_tp_id', 'status': 'success'}
        try:
            order = await self.exchange.create_order(symbol, 'market', side.lower(), amount, None, full_params)
            await self.update_weight(order)
            logging.info(f"สร้าง take profit order สำหรับ {symbol} สำเร็จ: ID {order['id']}")
            return order
        except Exception as e:
            logging.error(f"สร้าง take profit order สำหรับ {symbol} ล้มเหลว: {e}")
            return None

    async def update_order(self, symbol: str, order_id: str, params: Dict) -> Optional[Dict]:
        """อัพเดทคำสั่งที่มีอยู่"""
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการอัพเดท order ID {order_id} สำหรับ {symbol}")
            return {'status': 'success'}
        try:
            full_params = {**params, 'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()}
            response = await self.exchange.edit_order(order_id, symbol, None, None, None, None, full_params)
            await self.update_weight(response)
            logging.info(f"อัพเดท order ID {order_id} สำหรับ {symbol} สำเร็จ")
            return response
        except Exception as e:
            logging.error(f"อัพเดท order ID {order_id} ล้มเหลว: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """ยกเลิกคำสั่ง"""
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            logging.info(f"จำลองการยกเลิก order ID {order_id} สำหรับ {symbol}")
            return {'status': 'success'}
        try:
            params = {'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()}
            response = await self.exchange.cancel_order(order_id, symbol, params=params)
            await self.update_weight(response)
            logging.info(f"ยกเลิก order ID {order_id} สำหรับ {symbol} สำเร็จ")
            if symbol in self.trailing_orders and (order_id == self.trailing_orders[symbol].get('sl_order_id') or order_id == self.trailing_orders[symbol].get('tp_order_id')):
                del self.trailing_orders[symbol]
            return response
        except Exception as e:
            logging.error(f"ยกเลิก order ID {order_id} ล้มเหลว: {e}")
            return None

    async def check_symbol_exists(self, symbol: str) -> bool:
        """ตรวจสอบว่าเหรียญมีอยู่ใน Binance Futures"""
        await self.rate_limit_control()
        try:
            if self.markets is None:
                self.markets = await self.exchange.load_markets(params={'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()})
            return symbol in self.markets
        except Exception as e:
            logging.error(f"ตรวจสอบ symbol {symbol} ล้มเหลว: {e}")
            return False

    async def check_balance(self, asset: str = 'USDT') -> Dict:
        """ตรวจสอบยอดเงินในบัญชี Futures"""
        await self.rate_limit_control()
        if GlobalConfig.get('dry_run'):
            return {'free': 1000.0, 'total': 1000.0}
        try:
            balance = await self.exchange.fetch_balance(params={'type': 'future', 'recvWindow': 5000, 'timestamp': await self.get_adjusted_timestamp()})
            await self.update_weight(balance)
            return balance.get(asset, {'free': 0, 'total': 0})
        except Exception as e:
            logging.error(f"ตรวจสอบ balance ล้มเหลว: {e}")
            return {'free': 0, 'total': 0}

    async def predict_usage(self) -> float:
        """คาดการณ์การใช้งาน API ในอนาคต (60 วินาทีข้างหน้า)"""
        usage_history = [self.weight_used] * 60
        return sum(usage_history) / len(usage_history) if usage_history else 0
    # ===============================================
# websocket_manager.py
# คลาสจัดการ WebSocket สำหรับข้อมูลเรียลไทม์จาก Binance Futures
# ใช้ Config กลางจาก config.py เท่านั้น
# ===============================================

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install websockets aiohttp sqlite3 ccxt.async_support tenacity

import json
import asyncio
import time
import logging
import logging.handlers
import sqlite3
import websockets
from collections import deque
from datetime import datetime, timedelta
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt
#from config import GlobalConfig  # ดึง GlobalConfig จาก config.py

# การตั้งค่าระบบบันทึก log เฉพาะสำหรับคลาสนี้
logging.basicConfig(
    level=getattr(logging, GlobalConfig.get('log_level', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('websocketmanager.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

class WebSocketManager:
    def __init__(self, exchange=None, time_offset=0):
        """เริ่มต้น WebSocketManager
        Args:
            exchange: อินสแตนซ์ของ ccxt.async_support.binance จาก APIManager
            time_offset: ความแตกต่างของเวลา (มิลลิวินาที) จาก APIManager
        """
        self.url = GlobalConfig.get('ws_url')
        self.backup_url = GlobalConfig.get('ws_backup_url')
        self.data = {}  # เก็บข้อมูลเรียลไทม์จาก WebSocket
        self.running = False
        self.subscribed_symbols = set()
        self.reconnect_attempts = 0
        self.max_reconnects = GlobalConfig.get('max_reconnects')
        self.cache = {}  # Cache สำหรับข้อมูลล่าสุด
        self.all_usdt_pairs = []
        self.db_conn = sqlite3.connect(GlobalConfig.get('db_path'), timeout=10)
        cursor = self.db_conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS ws_data (symbol TEXT, timestamp REAL, close REAL, volume REAL, funding_rate REAL, depth REAL)")
        cursor.execute("CREATE TABLE IF NOT EXISTS historical_data (symbol TEXT, timestamp REAL, open REAL, high REAL, low REAL, close REAL, volume REAL)")
        self.db_conn.commit()
        self.balance_data = {'free': GlobalConfig.get('initial_balance', 100), 'locked': 0}  # เก็บยอดเงินเรียลไทม์
        self.position_data = {}  # เก็บข้อมูลตำแหน่งเรียลไทม์
        self.exchange = exchange  # อ้างอิง exchange จาก APIManager
        self.time_offset = time_offset  # ใช้ time_offset จาก APIManager
        self.multi_tf_data = {tf: deque(maxlen=1000) for tf in GlobalConfig.get('multi_tf_list', [])}  # เก็บ OHLCV aggregate ตาม timeframe
        self.last_aggregate_time = {tf: time.time() for tf in GlobalConfig.get('multi_tf_list', [])}  # เวลา aggregate ล่าสุดต่อ timeframe
        self.listen_key = None  # สำหรับ user stream

    async def create_listen_key(self):
        """สร้าง listenKey สำหรับ user stream"""
        if self.listen_key or not self.exchange:
            return self.listen_key
        try:
            response = await self.exchange.private_futures_start_user_stream()
            self.listen_key = response['listenKey']
            logging.info(f"สร้าง listenKey สำเร็จ: {self.listen_key[:10]}...")
            return self.listen_key
        except Exception as e:
            logging.error(f"สร้าง listenKey ล้มเหลว: {e}")
            return None

    async def keep_alive_loop(self):
        """ลูป keep alive user stream ทุก 30 นาที"""
        while self.running:
            await asyncio.sleep(1800)  # 30 นาที
            if self.listen_key and self.exchange:
                try:
                    await self.exchange.private_futures_keep_alive_user_stream({'listenKey': self.listen_key})
                    logging.debug("keep alive user stream สำเร็จ")
                except Exception as e:
                    logging.error(f"keep alive user stream ล้มเหลว: {e}")

    async def fetch_all_usdt_pairs(self):
        """ดึงรายการคู่เหรียญ USDT ทั้งหมดจาก Binance Futures"""
        try:
            markets = await self.exchange.load_markets()
            self.all_usdt_pairs = [s for s in markets if s.endswith('USDT') and markets[s].get('swap', False)]
            logging.info(f"ดึงคู่ USDT ทั้งหมด: {len(self.all_usdt_pairs)} เหรียญ")
        except Exception as e:
            logging.error(f"ดึงคู่ USDT ล้มเหลว: {e}")
            self.all_usdt_pairs = ['BTCUSDT', 'ETHUSDT']  # ค่า default

    async def load_historical_data(self, symbol, years=GlobalConfig.get('historical_years', 5)):
        """โหลดข้อมูลย้อนหลังสำหรับโหมดจำลอง"""
        years_ago = datetime.utcnow() - timedelta(days=365 * years)
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT timestamp, open, high, low, close, volume FROM historical_data WHERE symbol=? AND timestamp>=? ORDER BY timestamp ASC",
            (symbol, years_ago.timestamp())
        )
        data = cursor.fetchall()
        if len(data) < 1000:  # สมมติ limit สำหรับ historical
            try:
                klines = await self.exchange.fetch_ohlcv(symbol, '1h', since=int(years_ago.timestamp() * 1000), limit=1000)
                for kline in klines:
                    timestamp, open_p, high, low, close, volume = kline
                    self.db_conn.execute(
                        "INSERT OR IGNORE INTO historical_data (symbol, timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (symbol, timestamp / 1000, open_p, high, low, close, volume)
                    )
                self.db_conn.commit()
                logging.info(f"โหลดข้อมูลย้อนหลัง {years} ปีสำหรับ {symbol} สำเร็จ")
                cursor.execute(
                    "SELECT timestamp, open, high, low, close, volume FROM historical_data WHERE symbol=? ORDER BY timestamp ASC",
                    (symbol,)
                )
                data = cursor.fetchall()
            except Exception as e:
                logging.error(f"ดึงข้อมูลย้อนหลังสำหรับ {symbol} ล้มเหลว: {e}")
                data = [(time.time() - i*3600, 10000, 10100, 9900, 10000, 100) for i in range(1000)]  # จำลองข้อมูล
        return data

    async def aggregate_to_timeframe(self, symbol, tf, new_data):
        """รวมข้อมูลเรียลไทม์เป็น OHLCV ตาม timeframe"""
        tf_seconds = self._tf_to_seconds(tf)
        current_time = time.time()
        if current_time - self.last_aggregate_time[tf] >= tf_seconds:
            close = new_data.get('close', 0)
            volume = new_data.get('volume', 0)
            open_p = self.multi_tf_data[tf][-1][1] if self.multi_tf_data[tf] else close
            high = max(open_p, close)
            low = min(open_p, close)
            self.multi_tf_data[tf].append((current_time, open_p, high, low, close, volume))
            self.last_aggregate_time[tf] = current_time
            logging.debug(f"Aggregated data for {symbol} in {tf}")

    def _tf_to_seconds(self, tf):
        """แปลง timeframe เป็นวินาที"""
        units = {'m': 60, 'h': 3600, 'd': 86400}
        num = int(''.join(filter(str.isdigit, tf)))
        unit = tf[-1]
        return num * units.get(unit, 60)

    async def update_symbols(self, symbols):
        """อัพเดทรายการ symbols ที่สมัครรับข้อมูล"""
        new_symbols = [s.lower() + '@ticker' for s in symbols]
        if set(new_symbols) != self.subscribed_symbols:
            self.subscribed_symbols = set(new_symbols[:1024])  # Binance limit 1024 streams
            if self.running:
                await self.resubscribe()

    async def resubscribe(self, websocket=None):
        """สมัครรับข้อมูล WebSocket ใหม่"""
        if websocket:
            await self.create_listen_key()
            if self.listen_key:
                params = [f"{self.listen_key}@arr"] + list(self.subscribed_symbols)
            else:
                params = list(self.subscribed_symbols)
            await websocket.send(json.dumps({
                'method': 'SUBSCRIBE',
                'params': params,
                'id': 1
            }))
            logging.info(f"สมัครรับข้อมูลใหม่: {len(self.subscribed_symbols)} streams")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=GlobalConfig.get('reconnect_delay_max', 60)), stop=stop_after_attempt(GlobalConfig.get('max_reconnects', 10)))
    async def start(self, symbols):
        """เริ่มต้น WebSocket และรับข้อมูลเรียลไทม์"""
        if not self.all_usdt_pairs:
            await self.fetch_all_usdt_pairs()
        await self.update_symbols(symbols)
        self.running = True
        asyncio.create_task(self.keep_alive_loop())
        urls = [self.url, self.backup_url]
        current_url_idx = 0
        while self.running:
            try:
                async with websockets.connect(urls[current_url_idx]) as websocket:
                    await self.resubscribe(websocket)
                    self.reconnect_attempts = 0
                    while self.running:
                        message = await asyncio.wait_for(websocket.recv(), timeout=GlobalConfig.get('ws_timeout', 10))
                        data = json.loads(message)
                        await self._handle_message(data, websocket)
            except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                self.reconnect_attempts += 1
                logging.warning(f"WebSocket ล้มเหลว: {e}, พยายามใหม่ครั้งที่ {self.reconnect_attempts}")
                if self.reconnect_attempts >= self.max_reconnects:
                    logging.error("WebSocket ล้มเหลวเกินจำกัด ใช้ข้อมูลสำรอง")
                    await self.use_fallback_data(symbols)
                    break
                await asyncio.sleep(min(5 * self.reconnect_attempts, GlobalConfig.get('reconnect_delay_max', 60)))
                current_url_idx = (current_url_idx + 1) % len(urls)
            except Exception as e:
                logging.error(f"ข้อผิดพลาดใน WebSocket: {e}")
                break

    async def stop(self):
        """หยุด WebSocket และปิดฐานข้อมูลอย่างปลอดภัย"""
        if self.listen_key and self.exchange:
            try:
                await self.exchange.private_futures_close_user_stream({'listenKey': self.listen_key})
                logging.info("ปิด user stream สำเร็จ")
            except Exception as e:
                logging.error(f"ปิด user stream ล้มเหลว: {e}")
        self.running = False
        self.db_conn.close()
        logging.info("หยุด WebSocket และปิดฐานข้อมูลอย่างปลอดภัย")

    async def _handle_message(self, data, websocket):
        """จัดการข้อความจาก WebSocket"""
        if 'ping' in data:
            await websocket.send(json.dumps({'pong': data['ping']}))
            logging.debug("ส่ง pong ตอบ ping จาก Binance")
        elif 'e' in data:
            if data['e'] == 'ACCOUNT_UPDATE':
                self._update_balance(data)
                self._update_position(data)
            elif data['e'] == 'ticker':
                symbol = data['s']
                new_data = {
                    'close': float(data.get('c', 0)),
                    'volume': float(data.get('v', 0)),
                    'timestamp': (data.get('E', int(time.time() * 1000)) + self.time_offset) / 1000,
                    'funding_rate': 0.0001,  # Default เนื่องจาก ticker ไม่มี funding_rate
                    'depth': float(data.get('a', 0)) - float(data.get('b', 0))  # Spread จาก ask - bid
                }
                self._update_data(symbol, new_data)
                await self.save_to_sqlite(new_data, symbol)
                for tf in GlobalConfig.get('multi_tf_list', []):
                    await self.aggregate_to_timeframe(symbol, tf, new_data)

    def _update_balance(self, data):
        """อัพเดทยอดเงินจาก WebSocket"""
        balances = data.get('a', {}).get('B', [])
        for bal in balances:
            if bal.get('a') == 'USDT':
                self.balance_data = {
                    'free': float(bal.get('wb', 0)),
                    'locked': float(bal.get('cw', 0)) - float(bal.get('wb', 0))
                }
                logging.debug(f"อัพเดทยอดเงิน USDT: free={self.balance_data['free']}, locked={self.balance_data['locked']}")
                break

    def _update_position(self, data):
        """อัพเดทข้อมูลตำแหน่งจาก WebSocket"""
        positions = data.get('a', {}).get('P', [])
        for pos in positions:
            symbol = pos.get('s')
            if symbol and float(pos.get('pa', 0)) != 0:
                leverage = GlobalConfig.get('max_leverage_per_symbol', {}).get(symbol, 125)
                self.position_data[symbol] = {
                    'size': float(pos.get('pa', 0)),
                    'entryPrice': float(pos.get('ep', 0)),
                    'leverage': leverage,
                    'marginMode': pos.get('mt', 'cross')
                }
                logging.debug(f"อัพเดทตำแหน่ง {symbol}: size={self.position_data[symbol]['size']}, leverage={leverage}")

    def _update_data(self, symbol, new_data):
        """อัพเดทข้อมูลเรียลไทม์ใน cache"""
        self.data[symbol] = new_data
        self.cache[symbol] = self.data[symbol]
        if len(self.cache) > GlobalConfig.get('cache_size_max', 1000):
            self.cache.pop(next(iter(self.cache)))

    async def save_to_sqlite(self, data, symbol):
        """บันทึกข้อมูล WebSocket ลง SQLite"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "INSERT INTO ws_data (symbol, timestamp, close, volume, funding_rate, depth) VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, data['timestamp'], data['close'], data['volume'], data['funding_rate'], data['depth'])
        )
        limit = GlobalConfig.get('data_retention_limit', 100)
        cursor.execute(
            "DELETE FROM ws_data WHERE symbol=? AND timestamp NOT IN (SELECT timestamp FROM ws_data WHERE symbol=? ORDER BY timestamp DESC LIMIT ?)",
            (symbol, symbol, limit)
        )
        self.db_conn.commit()

    async def fetch_backup_data(self, symbol):
        """ดึงข้อมูลสำรองจาก SQLite"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            "SELECT close, volume, funding_rate, depth FROM ws_data WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",
            (symbol, GlobalConfig.get('data_retention_limit', 100))
        )
        data = cursor.fetchall()
        if data:
            logging.info(f"ดึงข้อมูลสำรองสำหรับ {symbol}: {len(data)} แถว")
            return data[::-1]
        return []

    async def use_fallback_data(self, symbols):
        """ใช้ข้อมูลสำรองเมื่อ WebSocket ล้มเหลว"""
        for symbol in symbols:
            backup_data = await self.fetch_backup_data(symbol)
            if backup_data:
                last_data = backup_data[-1]
                self.data[symbol] = {
                    'close': last_data[0],
                    'volume': last_data[1],
                    'timestamp': time.time(),
                    'funding_rate': last_data[2],
                    'depth': last_data[3]
                }
            else:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    self.data[symbol] = {
                        'close': ticker['last'],
                        'volume': ticker.get('quoteVolume', 0),
                        'timestamp': time.time(),
                        'funding_rate': 0.0001,
                        'depth': (ticker.get('ask', 0) - ticker.get('bid', 0))
                    }
                except Exception as e:
                    logging.error(f"ดึง ticker สำรองสำหรับ {symbol} ล้มเหลว: {e}")
                    self.data[symbol] = {
                        'close': 10000,
                        'volume': 100,
                        'timestamp': time.time(),
                        'funding_rate': 0.0001,
                        'depth': 0
                    }
            logging.info(f"ใช้ข้อมูลสำรองสำหรับ {symbol}: ราคา {self.data[symbol]['close']}")

    def get_latest_price(self, symbol):
        """ดึงราคาล่าสุดของเหรียญ"""
        default_price = 10000
        return self.cache.get(symbol, self.data.get(symbol, {})).get('close', default_price)

    def get_latest_balance(self):
        """ดึงยอดเงินล่าสุด"""
        return self.balance_data.get('free', 0)

    async def prefetch_data(self, symbols, timeframes):
        """ดึงข้อมูลล่วงหน้าสำหรับ timeframe ที่ระบุ"""
        for symbol in symbols:
            for tf in timeframes:
                try:
                    klines = await self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=100)
                    for kline in klines:
                        timestamp, open_p, high, low, close, volume = kline
                        self.multi_tf_data[tf].append((timestamp / 1000, open_p, high, low, close, volume))
                    logging.debug(f"ดึงข้อมูลล่วงหน้าสำหรับ {symbol} ใน {tf} สำเร็จ")
                except Exception as e:
                    logging.error(f"ดึงข้อมูลล่วงหน้าสำหรับ {symbol} ใน {tf} ล้มเหลว: {e}")

    async def check_symbol_exists(self, symbol):
        """ตรวจสอบว่าเหรียญมีอยู่ในรายการ USDT pairs"""
        return symbol in self.all_usdt_pairs
    # คลาสที่ 3: SSD

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch

import torch
import torch.nn as nn
import numpy as np

class SSD(nn.Module):
    """
    โมเดล State Space Dynamics สำหรับทำนายสถานะถัดไป
    การใช้งาน: ใช้ในการทำนายสถานะถัดไปในระบบการเทรด โดยปรับ learning rate ตาม volatility
    Config ที่เกี่ยวข้อง: 'min_volatility_threshold' สำหรับปรับ adaptive_lr_factor ใน method train
    """
    def __init__(self, input_dim):
        """
        เริ่มต้นโมเดล SSD
        Args:
            input_dim (int): ขนาดของ input vector (จำนวน features)
        """
        super(SSD, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.ReLU()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.01)
        self.adaptive_lr_factor = 1.0

    def forward(self, x):
        """
        ประมวลผล input ผ่าน encoder และ decoder
        Args:
            x (torch.Tensor): Input tensor ขนาด (batch_size, input_dim)
        Returns:
            torch.Tensor: Output tensor ที่ทำนายสถานะถัดไป
        """
        x = x.to(self.device)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def train(self, state_batch, next_state_batch, volatility=None):
        """
        ฝึกโมเดล SSD ด้วยข้อมูล state และ next_state
        Args:
            state_batch (np.ndarray): Batch ของสถานะปัจจุบัน
            next_state_batch (np.ndarray): Batch ของสถานะถัดไป
            volatility (float, optional): ความผันผวนสำหรับปรับ learning rate
        Returns:
            float: ค่า loss จากการฝึก
        """
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reconstructed = self.forward(state_batch)
        loss = nn.MSELoss()(reconstructed, next_state_batch)
        if volatility is not None:
            self.adaptive_lr_factor = min(1.0, max(0.1, volatility / GlobalConfig.get('min_volatility_threshold')))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001 * self.adaptive_lr_factor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    # คลาสที่ 4: QNetworkTransformer

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch transformers

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from collections import deque
import logging
import logging.handlers

# การตั้งค่าระบบบันทึก log เฉพาะสำหรับคลาสนี้
logging.basicConfig(
    level=getattr(logging, GlobalConfig.get('log_level')),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('qnetworktransformer.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

class QNetworkTransformer(nn.Module):
    def __init__(self, input_dim, action_space_size, timesteps=10):
        """เริ่มต้น QNetworkTransformer
        Args:
            input_dim (int): ขนาดของ state vector (เช่น จำนวน indicator)
            action_space_size (int): จำนวน actions ที่เป็นไปได้ (เช่น buy, sell, hold)
            timesteps (int, optional): จำนวน timesteps สำหรับ input (default: 10)
        """
        super(QNetworkTransformer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.action_space_size = action_space_size
        self.timesteps = timesteps

        # ตรวจสอบความถูกต้องของ input
        if input_dim <= 0 or action_space_size <= 0 or timesteps <= 0:
            logging.error(f"input_dim ({input_dim}), action_space_size ({action_space_size}), หรือ timesteps ({timesteps}) ต้องมากกว่า 0")
            raise ValueError("input_dim, action_space_size, และ timesteps ต้องมากกว่าหรือเท่ากับ 1")

        # ปรับขนาดโมเดลอัตโนมัติ
        self.n_embd = min(128, max(64, input_dim * 2))  # ปรับ n_embd ตาม input_dim
        self.n_layer = 4 if input_dim > 50 else 2  # ลด layer หาก input_dim เล็ก
        self.n_head = 8 if input_dim > 50 else 4  # ลด head หาก input_dim เล็ก
        self.dropout_rate = 0.1 if input_dim > 100 else 0.05  # ลด dropout หากโมเดลเล็ก
        self.learning_rate = 0.0001 if input_dim > 50 else 0.0005  # ปรับ learning rate ตามขนาดโมเดล
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.confidence_window = 50  # ขนาด window สำหรับ confidence

        # สร้าง GPT2 configuration
        config = GPT2Config(
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_positions=self.timesteps
        )

        # สร้างชั้น neural network
        self.transformer = GPT2Model(config)
        self.fc1 = nn.Linear(input_dim, self.n_embd)  # ปรับ input ให้เข้ากับ transformer
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.n_embd * self.timesteps + self.n_embd, 128)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.q_output = nn.Linear(128, action_space_size)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # เก็บ confidence history และ loss history
        self.confidence = deque(maxlen=self.confidence_window)
        self.loss_history = deque(maxlen=100)  # เก็บ loss ล่าสุด 100 ค่า

        # ย้ายโมเดลไปยัง device
        self.to(self.device)
        logging.info(f"เริ่มต้น QNetworkTransformer: input_dim={input_dim}, action_space_size={action_space_size}, timesteps={timesteps}, device={self.device}")

    def forward(self, x):
        """Forward pass สำหรับทำนาย Q-values
        Args:
            x (torch.Tensor): Input tensor shape [batch_size, timesteps, input_dim]
        Returns:
            torch.Tensor: Q-values สำหรับแต่ละ action shape [batch_size, action_space_size]
        """
        # ตรวจสอบ input shape
        if x.dim() != 3 or x.size(1) != self.timesteps or x.size(2) != self.input_dim:
            logging.error(f"Input shape ไม่ถูกต้อง: ได้ {x.shape}, คาดหวัง [batch_size, {self.timesteps}, {self.input_dim}]")
            raise ValueError(f"Input shape ต้องเป็น [batch_size, {self.timesteps}, {self.input_dim}]")

        batch_size = x.size(0)
        x = x.to(self.device)

        # ปรับ input ให้เข้ากับ transformer
        x = torch.relu(self.fc1(x))  # [batch_size, timesteps, n_embd]
        x = self.dropout1(x)

        # ส่งผ่าน transformer
        transformer_out = self.transformer(inputs_embeds=x).last_hidden_state  # [batch_size, timesteps, n_embd]

        # รวมกับ fully connected layer
        flat_x = transformer_out.view(batch_size, -1)  # [batch_size, timesteps * n_embd]
        transformer_last = transformer_out[:, -1, :]  # [batch_size, n_embd]
        combined = torch.cat((flat_x, transformer_last), dim=1)  # [batch_size, timesteps * n_embd + n_embd]
        fc2_out = torch.relu(self.fc2(combined))  # [batch_size, 128]
        fc2_out = self.dropout2(fc2_out)
        q_values = self.q_output(fc2_out)  # [batch_size, action_space_size]

        return q_values

    def train_step(self, states, actions, rewards, next_states, dones):
        """ฝึกโมเดลด้วย batch ของ states, actions, rewards, next_states, และ dones
        Args:
            states (torch.Tensor): States shape [batch_size, timesteps, input_dim]
            actions (torch.Tensor): Action indices shape [batch_size]
            rewards (torch.Tensor): Rewards shape [batch_size]
            next_states (torch.Tensor): Next states shape [batch_size, timesteps, input_dim]
            dones (torch.Tensor): Done flags shape [batch_size]
        Returns:
            float: ค่า loss จากการฝึก
        """
        # ตรวจสอบ input types และ shapes
        if not all(isinstance(x, torch.Tensor) for x in [states, actions, rewards, next_states, dones]):
            logging.error("Inputs ต้องเป็น torch.Tensor")
            raise ValueError("Inputs ต้องเป็น torch.Tensor")

        if states.shape[1:] != (self.timesteps, self.input_dim) or next_states.shape[1:] != (self.timesteps, self.input_dim):
            logging.error(f"State shape ไม่ถูกต้อง: ได้ {states.shape}, คาดหวัง [batch_size, {self.timesteps}, {self.input_dim}]")
            raise ValueError(f"State shape ต้องเป็น [batch_size, {self.timesteps}, {self.input_dim}]")

        if actions.dim() != 1 or rewards.dim() != 1 or dones.dim() != 1 or actions.size(0) != states.size(0):
            logging.error("Actions, rewards, และ dones ต้องเป็น 1D tensors และมีขนาด batch เดียวกับ states")
            raise ValueError("Actions, rewards, และ dones ต้องเป็น 1D tensors และมีขนาด batch เดียวกับ states")

        # แปลง inputs ไปยัง device
        states = states.to(self.device)
        actions = actions.to(self.device, dtype=torch.long)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device, dtype=torch.float)

        # คำนวณ Q-values
        q_values = self.forward(states)  # [batch_size, action_space_size]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # คำนวณ target Q-values
        with torch.no_grad():
            next_q_values = self.forward(next_states)  # [batch_size, action_space_size]
            max_next_q_values = next_q_values.max(dim=1)[0]  # [batch_size]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values  # [batch_size]

        # คำนวณ loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # อัพเดท confidence และ loss history
        self.loss_history.append(loss.item())
        confidence = min(1.0 / (loss.item() + 1e-6), 1000.0)  # จำกัด confidence
        self.confidence.append(confidence)

        # ปรับ learning rate อัตโนมัติตาม loss history
        if len(self.loss_history) >= 10:
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            if avg_loss > 1.0:
                self.optimizer.param_groups[0]['lr'] = min(self.learning_rate * 1.5, 0.001)
                logging.debug(f"เพิ่ม learning rate เป็น {self.optimizer.param_groups[0]['lr']:.6f} เนื่องจาก loss สูง ({avg_loss:.4f})")
            elif avg_loss < 0.1:
                self.optimizer.param_groups[0]['lr'] = max(self.learning_rate * 0.5, 1e-5)
                logging.debug(f"ลด learning rate เป็น {self.optimizer.param_groups[0]['lr']:.6f} เนื่องจาก loss ต่ำ ({avg_loss:.4f})")

        logging.debug(f"Training step เสร็จสิ้น: loss={loss.item():.4f}, confidence={confidence:.4f}")
        return loss.item()

    def select_action(self, state, epsilon=None):
        """เลือก action ด้วย epsilon-greedy policy
        Args:
            state (torch.Tensor): State shape [1, timesteps, input_dim]
            epsilon (float, optional): Exploration rate หากไม่ระบุใช้ self.epsilon
        Returns:
            int: Action index
        """
        epsilon = epsilon if epsilon is not None else self.epsilon

        if state.dim() == 2:
            state = state.unsqueeze(0)  # เพิ่ม batch dimension
        if state.shape[1:] != (self.timesteps, self.input_dim):
            logging.error(f"State shape ไม่ถูกต้อง: ได้ {state.shape}, คาดหวัง [1, {self.timesteps}, {self.input_dim}]")
            raise ValueError(f"State shape ต้องเป็น [1, {self.timesteps}, {self.input_dim}]")

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.action_space_size, (1,)).item()
            logging.debug(f"เลือก action แบบสุ่ม: {action}")
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.forward(state)  # [1, action_space_size]
                action = q_values.argmax(dim=1).item()
                logging.debug(f"เลือก action จาก Q-values: {action}")

        return action

    def save_model(self, path):
        """บันทึกโมเดลไปยังไฟล์
        Args:
            path (str): เส้นทางไฟล์สำหรับบันทึก
        """
        try:
            torch.save(self.state_dict(), path)
            logging.info(f"บันทึกโมเดลไปที่ {path}")
        except Exception as e:
            logging.error(f"บันทึกโมเดลล้มเหลว: {e}")

    def load_model(self, path):
        """โหลดโมเดลจากไฟล์
        Args:
            path (str): เส้นทางไฟล์สำหรับโหลด
        """
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
            logging.info(f"โหลดโมเดลจาก {path}")
        except Exception as e:
            logging.error(f"โหลดโมเดลล้มเหลว: {e}")

    def check_input_compatibility(self, state):
        """ตรวจสอบความเข้ากันได้ของ state
        Args:
            state (torch.Tensor): State tensor
        Returns:
            bool: True หากเข้ากันได้, False พร้อม log หากไม่เข้ากันได้
        """
        expected_shape = (self.timesteps, self.input_dim) if state.dim() == 2 else (1, self.timesteps, self.input_dim)
        if state.shape[-2:] != (self.timesteps, self.input_dim):
            logging.warning(f"State shape ไม่ตรงกับที่กำหนด: ได้ {state.shape}, คาดหวัง {expected_shape}")
            return False
        logging.debug("State เข้ากันได้กับโมเดล")
        return True
    # คลาสที่ 5: EvoGAN

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch numpy

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# กำหนด TIMESTEPS จากต้นฉบับ (ใช้ใน input shape)
TIMESTEPS = 10

class EvoGAN:
    def __init__(self, input_dim, action_space_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim * TIMESTEPS, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(256, action_space_size)),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(action_space_size, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(self.device)
        
        self.gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=0.0002, weight_decay=0.01, betas=(0.0, 0.99))
        self.disc_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0002, weight_decay=0.01, betas=(0.0, 0.99))
        self.gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.gen_optimizer, T_max=GlobalConfig.get('nas_iterations'))
        self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.disc_optimizer, T_max=GlobalConfig.get('nas_iterations'))
        
        self.evo_population = []
        self.strategy_confidence = {}
        self.critic_iters = 5
        self.gp_lambda = 10.0  # คิดเองจาก best practices
        self.diversity_weight = 0.5  # คิดเองจาก best practices
        self.perturb_std = 0.05  # คิดเองจาก best practices
        self.current_iter = 0  # สำหรับปรับอัตโนมัติ

    def generate(self, state):
        state = torch.FloatTensor(state).to(self.device)
        strategy = self.generator(state.view(-1, TIMESTEPS * state.shape[-1]))
        confidence = -self.discriminator(strategy).mean().item()
        self.strategy_confidence[tuple(strategy.cpu().detach().numpy()[0])] = confidence
        return strategy

    def train(self, real_strategies, fake_strategies):
        real_strategies = torch.FloatTensor(real_strategies).to(self.device)
        fake_strategies = torch.FloatTensor(fake_strategies).to(self.device)
        
        disc_losses = []
        for _ in range(self.critic_iters):
            self.disc_optimizer.zero_grad()
            real_validity = self.discriminator(real_strategies)
            fake_validity = self.discriminator(fake_strategies.detach())
            alpha = torch.rand(real_strategies.size(0), 1).to(self.device)
            interpolates = (alpha * real_strategies + (1 - alpha) * fake_strategies.detach()).requires_grad_(True)
            disc_interpolates = self.discriminator(interpolates)
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones_like(disc_interpolates),
                                            create_graph=True, retain_graph=True)[0]
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
            disc_loss = -real_validity.mean() + fake_validity.mean() + gp
            disc_loss.backward()
            self.disc_optimizer.step()
            disc_losses.append(disc_loss.item())
        
        self.gen_optimizer.zero_grad()
        gen_validity = self.discriminator(fake_strategies)
        gen_loss = -gen_validity.mean()
        gen_loss.backward()
        self.gen_optimizer.step()
        
        self.gen_scheduler.step()
        self.disc_scheduler.step()
        
        # ปรับอัตโนมัติ: ถ้า gen_loss สูงเกิน threshold, เพิ่ม critic_iters
        if gen_loss.item() > 1.0 and self.critic_iters < 10:
            self.critic_iters += 1
        
        return np.mean(disc_losses), gen_loss.item()

    def evolve(self, strategies, rewards):
        strategies_np = np.array(strategies)
        diversity_scores = -np.sum(strategies_np * np.log(strategies_np + 1e-10), axis=1)
        fitness = np.array(rewards) + self.diversity_weight * diversity_scores
        self.evo_population = sorted(zip(strategies, fitness), key=lambda x: x[1], reverse=True)[:10]
        
        for _ in range(5):
            indices = np.random.choice(range(5), 2, replace=False)
            parent1 = torch.FloatTensor(self.evo_population[indices[0]][0]).to(self.device)
            parent2 = torch.FloatTensor(self.evo_population[indices[1]][0]).to(self.device)
            child = (parent1 + parent2) / 2
            child.requires_grad_(True)
            self.gen_optimizer.zero_grad()
            loss = -self.discriminator(child.unsqueeze(0)).mean()
            loss.backward()
            with torch.no_grad():
                child = child - 0.01 * child.grad
            child = child.detach().cpu().numpy() + np.random.normal(0, 0.01, size=child.shape)
            self.evo_population.append((child, 0))
        
        return [p[0] for p in sorted(self.evo_population, key=lambda x: x[1], reverse=True)[:10]]

    def search_architecture(self, state_dim, action_dim):
        population = self._generate_initial_population()
        supernet = self.generator.state_dict()
        for iter in range(GlobalConfig.get('nas_iterations')):
            self.current_iter = iter
            fitness = self._evaluate_population(population, supernet)
            population = self._evolve_population(population, fitness)
            if iter % 20 == 0 and iter > 0:
                self._grow_architecture()
        return population[0]

    def _generate_initial_population(self):
        base_dict = self.generator.state_dict()
        population = []
        for _ in range(20):
            new_dict = {k: v.clone() + torch.randn_like(v) * self.perturb_std for k, v in base_dict.items()}
            population.append(new_dict)
        return population

    def _evaluate_population(self, population, supernet):
        fitness = []
        batch_size = 32 + (self.current_iter // 10)  # ปรับ batch size อัตโนมัติตาม iter
        dummy_state = np.random.randn(batch_size, TIMESTEPS * state_dim) * np.random.uniform(0.01, 0.05)
        dummy_real = np.random.uniform(0, 1, (batch_size, action_dim))
        for arch in population:
            temp_gen = type(self.generator)()
            temp_gen.load_state_dict(arch)
            temp_gen.to(self.device)
            fake = temp_gen(torch.FloatTensor(dummy_state).to(self.device)).detach().cpu().numpy()
            with torch.no_grad():
                super_fake = self.generator(torch.FloatTensor(dummy_state).to(self.device)).cpu().numpy()
            kd_loss = np.mean((fake - super_fake) ** 2)
            _, gen_loss = self.train(dummy_real, fake)
            entropy = -np.sum(fake * np.log(fake + 1e-10)) / len(fake)
            fit = - (gen_loss + kd_loss) + self.diversity_weight * entropy
            fitness.append(fit)
        return fitness

    def _evolve_population(self, population, fitness):
        sorted_idx = np.argsort(fitness)[::-1]
        sorted_pop = [population[i] for i in sorted_idx]
        new_pop = sorted_pop[:10]
        for _ in range(10):
            competitors = random.sample(range(len(sorted_pop)), 4)
            p1_idx = max(competitors[:2], key=lambda x: fitness[x])
            p2_idx = max(competitors[2:], key=lambda x: fitness[x])
            p1 = sorted_pop[p1_idx]
            p2 = sorted_pop[p2_idx]
            child = {}
            for k in p1:
                alpha = random.random()
                child[k] = alpha * p1[k] + (1 - alpha) * p2[k]
                child[k] += torch.randn_like(child[k]) * 0.02
                child[k] = torch.clamp(child[k], -1.0, 1.0)
            new_pop.append(child)
        return new_pop[:20]

    def _grow_architecture(self):
        for name, module in self.generator.named_modules():
            if isinstance(module, nn.Linear):
                if 'weight' in name:
                    module.out_features = int(module.out_features * 1.1)
                    # คลาสที่ 6: DDPG

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch numpy collections

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DDPG:
    def __init__(self, state_dim, action_dim, symbol):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = GlobalConfig.get('max_leverage_per_symbol', {}).get(symbol, 125)  # ซิงค์กับ global CONFIG จาก API
        self.min_action = GlobalConfig.get('min_leverage')  # ใช้ min_leverage จาก global CONFIG
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001, weight_decay=0.01)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001, weight_decay=0.01)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.confidence_history = deque(maxlen=50)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().detach().numpy() * self.max_action
        return np.clip(action, self.min_action, self.max_action)  # clip ระหว่าง min และ max leverage (ปรับอัตโนมัติ)

    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_actions = self.actor_target(next_state_batch)
        target_q = self.critic_target(torch.cat([next_state_batch, next_actions], dim=1))
        target_q = reward_batch + 0.99 * target_q.detach()
        current_q = self.critic(torch.cat([state_batch, action_batch], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actions_pred = self.actor(state_batch)
        actor_loss = -self.critic(torch.cat([state_batch, actions_pred], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)
        self.confidence_history.append(1 / (critic_loss.item() + 1e-6))
        # คลาสที่ 7: TFTWrapper

# ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch pandas numpy scikit-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from collections import deque
import logging
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TFTWrapper(nn.Module):
    def __init__(self, input_dim, forecast_horizon=60, d_model=64, nhead=4, num_layers=2):
        super(TFTWrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, input_dim)
        self.fusion_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.scaler = RobustScaler()
        self.confidence_scores = deque(maxlen=50)
        self.multi_tf_data = {}  # เพื่อเก็บข้อมูล multi-timeframe ถ้าจำเป็น
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.grad_scaler = GradScaler()
        self.to(self.device)

    def forward(self, x, tgt=None):
        # x: batch, seq, feature -> embed to d_model
        x = self.embedding(x)
        x = self.pos_encoder(x)
        memory = self.encoder(x)
        if tgt is None:
            tgt = torch.zeros(x.size(0), self.forecast_horizon, self.d_model).to(self.device)
        tgt = self.embedding(tgt) if tgt.shape[-1] == self.input_dim else tgt
        tgt = self.pos_encoder(tgt)
        out = self.decoder(tgt, memory)
        out = self.fc(out)
        return out

    def preprocess(self, data):
        df = pd.DataFrame(data, columns=['close', 'volume', 'RSI', 'MACD', 'RV', 'funding_rate', 'depth'])
        scaled_data = self.scaler.fit_transform(df)
        return torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)  # batch, seq, feat

    def preprocess_multi_tf(self, multi_tf_data):
        all_series = []
        for tf in GlobalConfig.get('multi_tf_list'):
            if tf in multi_tf_data and multi_tf_data[tf] is not None and not multi_tf_data[tf].empty:
                df = pd.DataFrame(multi_tf_data[tf], columns=['close', 'volume', 'RSI', 'MACD', 'RV', 'funding_rate', 'depth'])
                scaled_data = self.scaler.fit_transform(df)
                embedded = self.embedding(torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device))  # batch, seq, d_model
                all_series.append(embedded)
        if all_series:
            # Fuse with attention: stack to seq_len=num_tf, batch, d_model
            stacked = torch.cat(all_series, dim=1)  # batch, total_seq, d_model
            fused, _ = self.fusion_attention(stacked.transpose(0, 1), stacked.transpose(0, 1), stacked.transpose(0, 1))
            fused = fused.transpose(0, 1)  # back to batch, seq, d_model
            return fused
        return self.preprocess(np.zeros((10, 7)))

    def predict(self, state_data):
        if isinstance(state_data, dict):
            input_tensor = self.preprocess_multi_tf(state_data)
        else:
            input_tensor = self.preprocess(state_data)
        with torch.no_grad(), autocast():
            pred = self.forward(input_tensor)
        pred_values = pred.squeeze(0).cpu().numpy()
        # Improved confidence: use MC dropout for uncertainty
        self.train()  # enable dropout
        with torch.no_grad(), autocast():
            mc_preds = [self.forward(input_tensor).squeeze(0).cpu().numpy() for _ in range(10)]
        std = np.std(mc_preds, axis=0)
        confidence = np.mean(1 / (std + 1e-6))
        self.confidence_scores.append(confidence)
        self.eval()  # back to eval
        return pred_values

    def train(self, state_batch, target_batch):
        if isinstance(state_batch, dict):
            state_tensor = self.preprocess_multi_tf(state_batch)
            target_tensor = self.preprocess_multi_tf(target_batch)
        else:
            state_tensor = self.preprocess(state_batch)
            target_tensor = self.preprocess(target_batch)
        loss_fn = nn.MSELoss()
        with autocast():
            out = self.forward(state_tensor, tgt=target_tensor[:, :-self.forecast_horizon, :])
            # Adjust shapes
            if out.shape != target_tensor.shape:
                target_tensor = target_tensor[:, :out.shape[1], :]
            loss = loss_fn(out, target_tensor)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step(loss.item())

    def update_realtime(self, ws_data, symbol):
        if symbol in ws_data:
            latest_data = np.array([[ws_data[symbol]['close'], ws_data[symbol]['volume'], 0, 0, 0,
                                     ws_data[symbol]['funding_rate'], ws_data[symbol]['depth']]])
            # Online fine-tuning with small batch
            self.train(latest_data, latest_data)  # simple update
            logging.debug(f"อัพเดท TFT เรียลไทม์สำหรับ {symbol}")
            # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch-geometric

import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import numpy as np
from collections import deque

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ตรวจสอบ GPU อัตโนมัติ
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            if '5070 Ti' in prop.name:
                print(f"Detected RTX 5070 Ti with CUDA capability {prop.major}.{prop.minor}")
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.confidence = deque(maxlen=50)
        self.to(self.device)

    def forward(self, x, edge_index):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

    def train(self, graph_data):
        self.optimizer.zero_grad()
        out = self.forward(graph_data.x, graph_data.edge_index)
        loss = torch.nn.MSELoss()(out, graph_data.y)
        loss.backward()
        self.optimizer.step()
        self.confidence.append(1 / (loss.item() + 1e-6))
        return loss.item()
    # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch numpy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class MADRL:
    """
    Multi-Agent Deep Reinforcement Learning (MADRL) สำหรับการเทรดหลายเหรียญ
    ใช้ Actor-Critic architecture กับ multi-agent สำหรับจัดการ portfolio
    """

    def __init__(self, state_dim, action_dim, num_agents=None, symbols=None):
        """
        เริ่มต้นคลาส MADRL
        :param state_dim: int → ขนาดของ state (เช่น จำนวน features ต่อ agent)
        :param action_dim: int → ขนาดของ action (เช่น leverage หรือ position size)
        :param num_agents: int → จำนวน agent (default จาก CONFIG['madrl_agent_count'])
        :param symbols: list → รายการ symbols (เช่น ['BTCUSDT', 'ETHUSDT'])
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ตรวจสอบ GPU อัตโนมัติ
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            if '5070 Ti' in prop.name:
                print(f"Detected RTX 5070 Ti with CUDA capability {prop.major}.{prop.minor}")
        self.num_agents = min(num_agents or CONFIG['madrl_agent_count'], CONFIG['madrl_agent_count'])
        self.symbols = symbols or []
        self.max_actions = [CONFIG['max_leverage_per_symbol'].get(s, 125) for s in self.symbols]
        self.actors = [nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device) for _ in range(self.num_agents)]
        self.actor_targets = [nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device) for _ in range(self.num_agents)]
        self.critic = nn.Sequential(
            nn.Linear(state_dim * self.num_agents + action_dim * self.num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim * self.num_agents + action_dim * self.num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=0.0001) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        for i in range(self.num_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.confidence_history = deque(maxlen=50)

    def act(self, states):
        """
        สร้าง action จาก states ปัจจุบัน
        :param states: list of np.array → states สำหรับแต่ละ agent (shape: [num_agents, state_dim])
        :return: np.array → actions ที่ clip แล้ว (shape: [num_agents, action_dim])
        """
        if len(states) != self.num_agents:
            raise ValueError(f"จำนวน states ({len(states)}) ไม่ตรงกับ num_agents ({self.num_agents})")
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions = [self.actors[i](states_tensor[i]).cpu().detach().numpy() * self.max_actions[i] for i in range(self.num_agents)]
        actions_array = np.array(actions)
        return np.clip(actions_array, CONFIG['min_leverage'], self.max_actions)

    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        """
        ฝึกโมเดลด้วย batch ของข้อมูล
        :param state_batch: np.array → states (shape: [batch_size, num_agents, state_dim])
        :param action_batch: np.array → actions (shape: [batch_size, num_agents, action_dim])
        :param reward_batch: np.array → rewards (shape: [batch_size, 1])
        :param next_state_batch: np.array → next states (shape: [batch_size, num_agents, state_dim])
        """
        if len(state_batch) != len(action_batch) or len(state_batch) != len(reward_batch) or len(state_batch) != len(next_state_batch):
            raise ValueError("Batch ขนาดไม่ตรงกัน")
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        
        # คำนวณ next_actions จาก target actors
        next_actions_list = [self.actor_targets[i](next_state_batch[:, i, :]) for i in range(self.num_agents)]
        next_actions = torch.stack(next_actions_list, dim=1)  # [batch_size, num_agents, action_dim]
        
        # เตรียม input สำหรับ critic (flatten across agents)
        critic_input = torch.cat([state_batch.view(state_batch.size(0), -1), 
                                  action_batch.view(action_batch.size(0), -1)], dim=1)
        next_critic_input = torch.cat([next_state_batch.view(next_state_batch.size(0), -1), 
                                       next_actions.view(next_actions.size(0), -1)], dim=1)
        
        # คำนวณ target Q-value
        target_q = reward_batch + 0.99 * self.critic_target(next_critic_input)
        current_q = self.critic(critic_input)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        
        # อัพเดท critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # อัพเดท actors
        for i in range(self.num_agents):
            actions_pred = self.actors[i](state_batch[:, i, :])  # [batch_size, action_dim]
            # สร้าง input สำหรับ actor loss (ใช้ actions จาก agent นี้และ actions อื่นจาก batch)
            # สำหรับ simplicity, ใช้ average หรือ full concat แต่ปรับให้ถูกต้อง
            # ที่นี่ใช้ concat กับ dummy actions สำหรับ agents อื่น (ปรับจาก original)
            other_actions = action_batch[:, :i, :].view(action_batch.size(0), -1)
            this_actions_exp = actions_pred.unsqueeze(1).expand(-1, i, -1).view(action_batch.size(0), -1) if i > 0 else torch.zeros(action_batch.size(0), 0).to(self.device)
            post_actions = action_batch[:, i+1:, :].view(action_batch.size(0), -1)
            actor_input = torch.cat([state_batch.view(state_batch.size(0), -1), 
                                     torch.cat([other_actions, this_actions_exp, post_actions], dim=1)], dim=1)
            actor_loss = -self.critic(actor_input).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        
        # Soft update targets
        tau = 0.001
        for i in range(self.num_agents):
            for t_param, param in zip(self.actor_targets[i].parameters(), self.actors[i].parameters()):
                t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
        for t_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
        
        self.confidence_history.append(1 / (critic_loss.item() + 1e-6))

    def update_symbols(self, new_symbols):
        """
        อัพเดท symbols และปรับ num_agents ใหม่
        :param new_symbols: list → รายการ symbols ใหม่
        """
        self.symbols = new_symbols
        old_num = self.num_agents
        self.num_agents = min(len(new_symbols), CONFIG['madrl_agent_count'])
        self.max_actions = [CONFIG['max_leverage_per_symbol'].get(s, 125) for s in self.symbols]
        
        # Trim actors ถ้า num_agents ลดลง
        if self.num_agents < old_num:
            self.actors = self.actors[:self.num_agents]
            self.actor_targets = self.actor_targets[:self.num_agents]
            self.actor_optimizers = self.actor_optimizers[:self.num_agents]
        # ถ้าเพิ่ม ต้องสร้างใหม่ (แต่ที่นี่ assume ไม่เพิ่มเกิน limit)
        elif self.num_agents > old_num:
            # สร้างเพิ่ม
            for _ in range(old_num, self.num_agents):
                actor = nn.Sequential(
                    nn.Linear(CONFIG.get('state_dim', 10), 256),  # assume default
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, CONFIG.get('action_dim', 1)),  # assume default
                    nn.Tanh()
                ).to(self.device)
                self.actors.append(actor)
                actor_target = nn.Sequential(
                    nn.Linear(CONFIG.get('state_dim', 10), 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, CONFIG.get('action_dim', 1)),
                    nn.Tanh()
                ).to(self.device)
                self.actor_targets.append(actor_target)
                self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=0.0001))
                actor_target.load_state_dict(actor.state_dict())
            
            # อัพเดท critic ถ้าจำเป็น (recreate ถ้า dim เปลี่ยน)
            self._rebuild_critic(CONFIG.get('state_dim', 10), CONFIG.get('action_dim', 1))
    
    def _rebuild_critic(self, state_dim, action_dim):
        """Rebuild critic ถ้า num_agents เปลี่ยน"""
        input_dim = state_dim * self.num_agents + action_dim * self.num_agents
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_target = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def get_confidence(self):
        """
        ดึงค่า confidence ล่าสุด (1 / loss)
        :return: float → confidence score ล่าสุด
        """
        return np.mean(list(self.confidence_history)) if self.confidence_history else 0.0

    def save_model(self, path):
        """
        บันทึกโมเดล
        :param path: str → path สำหรับบันทึก
        """
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'actor_targets': [target.state_dict() for target in self.actor_targets],
            'critic_target': self.critic_target.state_dict(),
            'symbols': self.symbols,
            'num_agents': self.num_agents
        }, path)

    def load_model(self, path):
        """
        โหลดโมเดล
        :param path: str → path สำหรับโหลด
        """
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            if i < len(checkpoint['actors']):
                actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        for i, target in enumerate(self.actor_targets):
            if i < len(checkpoint['actor_targets']):
                target.load_state_dict(checkpoint['actor_targets'][i])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.symbols = checkpoint.get('symbols', self.symbols)
        self.num_agents = checkpoint.get('num_agents', self.num_agents)
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch numpy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MetaSelector(nn.Module):
    def __init__(self, input_dim):
        super(MetaSelector, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ตรวจสอบ GPU อัตโนมัติ
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            if '5070 Ti' in prop.name:
                print(f"Detected RTX 5070 Ti with CUDA capability {prop.major}.{prop.minor}")
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=CONFIG['maml_lr_outer'])
        self.confidence = {}

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def predict(self, state):
        with torch.no_grad():
            score = self.forward(torch.FloatTensor(state).to(self.device)).cpu().numpy()
            symbol_key = tuple(state)
            confidence = self.confidence.get(symbol_key, 1.0)
        return score[0] * confidence

    def train_few_shot(self, state_batch, reward_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        fast_weights = [p.clone() for p in self.parameters()]
        for _ in range(CONFIG['maml_steps']):
            pred = self.model(state_batch)
            loss = nn.MSELoss()(pred, reward_batch)
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            fast_weights = [w - CONFIG['maml_lr_inner'] * g for w, g in zip(fast_weights, grads)]
        for i, state in enumerate(state_batch):
            symbol_key = tuple(state.cpu().numpy())
            self.confidence[symbol_key] = 1 / (loss.item() + 1e-6)
        return fast_weights, loss.item()

    def train_meta(self, task_batch):
        meta_loss = 0
        for states, rewards in task_batch:
            fast_weights, loss = self.train_few_shot(states, rewards)
            pred = nn.Sequential(*self.model)(torch.FloatTensor(states).to(self.device))
            meta_loss += nn.MSELoss()(pred, torch.FloatTensor(rewards).to(self.device))
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        return meta_loss.item()
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch numpy bayesian-optimization pandas matplotlib seaborn darts torch-geometric scikit-learn

import torch
import numpy as np
from collections import deque
from bayes_opt import BayesianOptimization
import gc
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class UnifiedQuantumTrader:
    def __init__(self, input_dim, discrete_action_size=3, continuous_action_dim=2, num_symbols=CONFIG['madrl_agent_count']):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ตรวจสอบ GPU อัตโนมัติ
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            if '5070 Ti' in prop.name:
                print(f"Detected RTX 5070 Ti with CUDA capability {prop.major}.{prop.minor}")
        self.input_dim = input_dim
        self.discrete_action_size = discrete_action_size
        self.continuous_action_dim = continuous_action_dim
        self.qnt = QNetworkTransformer(input_dim, discrete_action_size)
        self.evogan = EvoGAN(input_dim, discrete_action_size)
        self.ddpg = DDPG(input_dim, continuous_action_dim, 'BTC/USDT')
        self.ssd = SSD(input_dim)
        self.tft = TFTWrapper(input_dim)
        self.gnn = GNN(input_dim)
        self.madrl = MADRL(input_dim, continuous_action_dim, num_symbols, symbols=['BTC/USDT']*num_symbols)
        self.meta_selector = MetaSelector(input_dim)
        self.replay_buffer = None  # จะเชื่อมใน main
        self.strategy_memory = []
        self.loss_history = {
            'qnt': [], 'ssd': [], 'evogan_disc': [], 'evogan_gen': [],
            'ddpg_actor': [], 'ddpg_critic': [], 'tft': [], 'gnn': [], 'madrl': [], 'meta': []
        }
        self.bayes_opt = BayesianOptimization(
            f=self._bayes_objective,
            pbounds={'qnt_w': (0, 1), 'evogan_w': (0, 1), 'tft_w': (0, 1), 'gnn_w': (0, 1), 'madrl_w': (0, 1)},
            random_state=42
        )
        self.model_weights = {'qnt_w': 0.2, 'evogan_w': 0.2, 'tft_w': 0.2, 'gnn_w': 0.2, 'madrl_w': 0.2}
        self.resource_manager = IntelligentResourceManager()  # เชื่อมใน main
        self.overfit_detector = {'val_loss': deque(maxlen=50), 'train_loss': deque(maxlen=50)}
        self.best_loss = float('inf')
        self.patience = 10
        self.wait = 0
        self.adaptive_symbol_selector = {}
        self.current_symbols = ['BTC/USDT'] * num_symbols
        self.multi_tf_data = {tf: deque(maxlen=1000) for tf in CONFIG['multi_tf_list']}
        self.risk_guardian = RiskGuardian()  # เชื่อมใน main
        self.scaler = torch.cuda.amp.GradScaler()

    def _bayes_objective(self, qnt_w, evogan_w, tft_w, gnn_w, madrl_w):
        total = qnt_w + evogan_w + tft_w + gnn_w + madrl_w
        if total == 0:
            return 0
        weights = {k: v / total for k, v in zip(['qnt_w', 'evogan_w', 'tft_w', 'gnn_w', 'madrl_w'], [qnt_w, evogan_w, tft_w, gnn_w, madrl_w])}
        if self.replay_buffer and self.replay_buffer.buffer:
            batch = self.replay_buffer.sample(32)
            if batch:
                states, discrete_actions, continuous_actions, rewards, _, _, _, _, _ = batch
                pred_discrete, pred_continuous = self._combine_predictions(states, weights)
                discrete_loss = np.mean((pred_discrete - discrete_actions) ** 2)
                continuous_loss = np.mean((pred_continuous - continuous_actions) ** 2)
                return -(discrete_loss + continuous_loss)
        return 0

    def _combine_predictions(self, state_data, weights):
        with torch.cuda.amp.autocast():
            q_values = self.qnt(torch.FloatTensor(state_data).to(self.device)).cpu().detach().numpy()
            evogan_strategies = self.evogan.generate(state_data).cpu().detach().numpy()
            tft_pred = self.tft.predict(state_data)
            gnn_pred = self.gnn.forward(torch.FloatTensor(state_data).to(self.device),
                                      self._create_graph(len(state_data))).cpu().detach().numpy()
            madrl_actions = self.madrl.act(state_data)
        model_confidences = {
            'qnt': np.mean(self.qnt.confidence) if self.qnt.confidence else 1.0,
            'evogan': np.mean(list(self.evogan.strategy_confidence.values())) if self.evogan.strategy_confidence else 1.0,
            'tft': np.mean(self.tft.confidence_scores) if self.tft.confidence_scores else 1.0,
            'gnn': np.mean(self.gnn.confidence) if self.gnn.confidence else 1.0,
            'madrl': np.mean(self.madrl.confidence_history) if self.madrl.confidence_history else 1.0
        }
        total_confidence = sum(model_confidences.values())
        dynamic_weights = {k: v / total_confidence * weights[k] for k, v in model_confidences.items()}
        discrete_pred = (dynamic_weights['qnt'] * q_values +
                        dynamic_weights['evogan'] * evogan_strategies +
                        dynamic_weights['tft'] * tft_pred[:, :self.discrete_action_size])
        continuous_pred = (dynamic_weights['madrl'] * madrl_actions +
                          dynamic_weights['gnn'] * gnn_pred[:, :self.continuous_action_dim])
        return discrete_pred, continuous_pred

    def _create_graph(self, num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                symbol_i = self.current_symbols[i % len(self.current_symbols)]
                symbol_j = self.current_symbols[j % len(self.current_symbols)]
                if symbol_i in ws_manager.data and symbol_j in ws_manager.data:
                    corr = np.corrcoef(
                        [ws_manager.data[symbol_i]['close']] * TIMESTEPS,
                        [ws_manager.data[symbol_j]['close']] * TIMESTEPS
                    )[0, 1] if ws_manager.data[symbol_i]['close'] and ws_manager.data[symbol_j]['close'] else 0
                    if abs(corr) > 0.5:
                        edges.append([i, j])
                        edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device) if edges else torch.tensor([[0, 0]], dtype=torch.long).t().to(self.device)
        return Data(x=torch.ones((num_nodes, self.input_dim)), edge_index=edge_index)

    def select_top_coins(self, all_symbols, ws_data, kpi_tracker):
        scores = {}
        multi_tf_states = self._aggregate_multi_tf_data(ws_data)
        for symbol in all_symbols:
            if symbol in ws_data:
                state = np.array([ws_data[symbol]['close'], ws_data[symbol]['volume'], 0, 0, 0,
                                ws_data[symbol]['funding_rate'], ws_data[symbol]['depth']])
                meta_score = self.meta_selector.predict(state)
                reward_pred = self.tft.predict(multi_tf_states.get(symbol, state.reshape(1, -1)))
                volatility = np.std(reward_pred)
                liquidity = ws_data[symbol]['volume']
                if liquidity >= CONFIG['liquidity_threshold']:
                    scores[symbol] = (meta_score * reward_pred.mean() * volatility * liquidity) / (1 + ws_data[symbol]['funding_rate'])
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        num_coins = max(CONFIG['min_coins_per_trade'], min(CONFIG['max_coins_per_trade'],
                        int(len(sorted_symbols) * (kpi_tracker.total_profit / CONFIG['target_kpi_daily'] + 0.5))))
        top_symbols = [s[0] for s in sorted_symbols[:num_coins]]
        self.current_symbols = top_symbols
        self.madrl.update_symbols(top_symbols)
        self.adaptive_symbol_selector = scores
        logging.info(f"เลือก {len(top_symbols)} เหรียญ: {top_symbols[:5]}...")
        return top_symbols

    def _aggregate_multi_tf_data(self, ws_data):
        multi_tf_states = {}
        for symbol in ws_data:
            state = np.array([ws_data[symbol]['close'], ws_data[symbol]['volume'], 0, 0, 0,
                            ws_data[symbol]['funding_rate'], ws_data[symbol]['depth']])
            for tf in CONFIG['multi_tf_list']:
                self.multi_tf_data[tf].append(state)
            multi_tf_states[symbol] = {tf: np.array(list(self.multi_tf_data[tf])[-10:]) for tf in CONFIG['multi_tf_list']}
        return multi_tf_states

    def predict(self, state_data):
        self.resource_manager.adjust_resources(self)
        discrete_pred, continuous_pred = self._combine_predictions(state_data, self.model_weights)
        return discrete_pred, continuous_pred

    async def train(self, states, discrete_actions, continuous_actions, rewards, next_states):
        batch_size = min(len(states), self.resource_manager.model_batch_sizes['qnt'])
        if batch_size < 1:
            return
        states = np.array(states[:batch_size])
        discrete_actions = np.array(discrete_actions[:batch_size])
        continuous_actions = np.array(continuous_actions[:batch_size])
        rewards = np.array(rewards[:batch_size])
        next_states = np.array(next_states[:batch_size])
        volatility = np.mean([self.replay_buffer.buffer[-1][-2] for _ in range(min(10, len(self.replay_buffer.buffer)))
                            if self.replay_buffer.buffer[-1][-2] is not None]) if self.replay_buffer.buffer else 0.01
        lr_factor = min(1.0, max(0.1, volatility / CONFIG['min_volatility_threshold']))
        for opt in [self.qnt.optimizer, self.ddpg.actor_optimizer, self.ddpg.critic_optimizer]:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.0001 * lr_factor
        val_size = int(batch_size * 0.2)
        train_size = batch_size - val_size
        idx = np.random.permutation(batch_size)
        train_idx, val_idx = idx[:train_size], idx[train_size:]
        train_states, val_states = states[train_idx], states[val_idx]
        train_discrete, val_discrete = discrete_actions[train_idx], discrete_actions[val_idx]
        train_continuous, val_continuous = continuous_actions[train_idx], continuous_actions[val_idx]
        train_rewards, val_rewards = rewards[train_idx], rewards[val_idx]
        train_next_states, val_next_states = next_states[train_idx], next_states[val_idx]

        with torch.cuda.amp.autocast():
            qnt_loss = self.qnt.train_step(train_states, train_discrete, train_rewards, train_next_states, torch.zeros(len(train_states)))
            self.ddpg.train(train_states, train_continuous, train_rewards, train_next_states)
            ssd_loss = self.ssd.train(train_states, train_next_states, volatility)
            disc_loss, gen_loss = self.evogan.train(train_discrete, self.evogan.generate(train_states).cpu().detach().numpy())
            self.tft.train(train_states, train_next_states)
            self.madrl.train(train_states, train_continuous, train_rewards, train_next_states)
            meta_loss = self.meta_selector.train_meta([(train_states, train_rewards)])

        val_q_values = self.qnt(torch.FloatTensor(val_states).to(self.device)).cpu().detach().numpy()
        val_loss = np.mean((val_q_values - val_discrete) ** 2)
        train_loss = np.mean((self.qnt(torch.FloatTensor(train_states).to(self.device)).cpu().detach().numpy() - train_discrete) ** 2)
        self.overfit_detector['val_loss'].append(val_loss)
        self.overfit_detector['train_loss'].append(train_loss)

        if len(self.overfit_detector['val_loss']) > 10 and np.mean(self.overfit_detector['val_loss']) > np.mean(self.overfit_detector['train_loss']) * 1.5:
            logging.warning(f"ตรวจพบ overfitting: val_loss={np.mean(self.overfit_detector['val_loss']):.4f}, train_loss={np.mean(self.overfit_detector['train_loss']):.4f}")
            for model in [self.qnt, self.ddpg.actor, self.ddpg.critic, self.ssd]:
                for param_group in model.optimizer.param_groups:
                    param_group['weight_decay'] *= 1.2
            for layer in [self.qnt.dropout1, self.qnt.dropout2]:
                layer.p = min(layer.p + 0.1, 0.5)
            logging.info("ปรับ regularization เพื่อแก้ overfitting")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info("Early stopping ทำงาน หยุดฝึกเพื่อป้องกัน overfitting")
                return

        self.loss_history['qnt'].append(train_loss)
        self.loss_history['ssd'].append(ssd_loss)
        self.loss_history['evogan_disc'].append(disc_loss)
        self.loss_history['evogan_gen'].append(gen_loss)
        self.loss_history['ddpg_critic'].append(np.mean((self.ddpg.critic(torch.FloatTensor(np.concatenate([train_states, train_continuous], axis=1)).to(self.device)).cpu().detach().numpy() - train_rewards) ** 2))
        self.loss_history['tft'].append(np.mean((self.tft.predict(train_states) - train_next_states) ** 2))
        self.loss_history['madrl'].append(np.mean((self.madrl.critic(torch.FloatTensor(np.concatenate([train_states.flatten(), train_continuous.flatten()])).to(self.device)).cpu().detach().numpy() - train_rewards) ** 2))
        self.loss_history['meta'].append(meta_loss)

    async def evolve(self, state_data, reward, volatility):
        strategies = self.evogan.generate(state_data).cpu().detach().numpy()
        self.strategy_memory.extend(strategies)
        if len(self.strategy_memory) >= 10:
            evolved_strategies = self.evogan.evolve(self.strategy_memory, [reward] * len(self.strategy_memory))
            self.strategy_memory = evolved_strategies[:10]

    async def adversarial_train(self, states):
        noise = np.random.normal(0, 0.1, states.shape)
        adv_states = states + noise
        adv_q_values = self.qnt(torch.FloatTensor(adv_states).to(self.device)).cpu().detach().numpy()
        adv_continuous = self.ddpg.act(adv_states)
        batch = self.replay_buffer.sample(32)
        if batch:
            orig_states, discrete_actions, continuous_actions, rewards, next_states, _, _, _, _ = batch
            await self.train(np.concatenate([orig_states, adv_states]),
                           np.concatenate([discrete_actions, discrete_actions]),
                           np.concatenate([continuous_actions, adv_continuous]),
                           np.concatenate([rewards, rewards * 0.5]),
                           np.concatenate([next_states, next_states]))
                           # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install sqlite3 zlib pickle numpy sklearn

import sqlite3
import zlib
import pickle
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
import logging

class ReplayBuffer:
    def __init__(self, db_path='replay_buffer.db', capacity=10000):
        self.db_path = db_path
        self.buffer = deque(maxlen=capacity)
        self.db_conn = sqlite3.connect(self.db_path, timeout=10)
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY, state BLOB, discrete_action INTEGER, continuous_action BLOB, reward REAL, next_state BLOB, gnn_embedding BLOB, tft_pred BLOB, atr REAL, multi_tf_data BLOB)")
        self.db_size = 0
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.load_from_db()

    def add(self, state, discrete_action, continuous_action, reward, next_state, gnn_embedding=None, tft_pred=None, atr=None, multi_tf_data=None):
        state_blob = zlib.compress(pickle.dumps(state))
        continuous_action_blob = zlib.compress(pickle.dumps(continuous_action))
        next_state_blob = zlib.compress(pickle.dumps(next_state))
        gnn_embedding_blob = zlib.compress(pickle.dumps(gnn_embedding)) if gnn_embedding is not None else None
        tft_pred_blob = zlib.compress(pickle.dumps(tft_pred)) if tft_pred is not None else None
        multi_tf_data_blob = zlib.compress(pickle.dumps(multi_tf_data)) if multi_tf_data is not None else None
        features = np.concatenate([state.flatten(), [discrete_action], continuous_action, [reward]])
        if len(self.buffer) > 50 and self.anomaly_detector.predict([features])[0] == -1:
            logging.warning(f"ตรวจพบ anomaly ในข้อมูล: reward={reward}, ปรับลดน้ำหนัก")
            reward *= 0.5
        self.buffer.append((state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data))
        with self.db_conn:
            self.db_conn.execute("INSERT INTO experiences (state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (state_blob, discrete_action, continuous_action_blob, reward, next_state_blob, gnn_embedding_blob, tft_pred_blob, atr, multi_tf_data_blob))
            self.db_size += 1
        if self.db_size % 100 == 0:
            self.fit_anomaly_detector()

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([x[0] for x in batch])
        discrete_actions = np.array([x[1] for x in batch])
        continuous_actions = np.array([x[2] for x in batch])
        rewards = np.array([x[3] for x in batch])
        next_states = np.array([x[4] for x in batch])
        gnn_embeddings = np.array([x[5] for x in batch if x[5] is not None], dtype=object)
        tft_preds = np.array([x[6] for x in batch if x[6] is not None], dtype=object)
        atrs = np.array([x[7] for x in batch if x[7] is not None])
        multi_tf_data = np.array([x[8] for x in batch if x[8] is not None], dtype=object)
        return states, discrete_actions, continuous_actions, rewards, next_states, gnn_embeddings, tft_preds, atrs, multi_tf_data

    def load_from_db(self):
        with self.db_conn:
            cursor = self.db_conn.execute("SELECT state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data FROM experiences ORDER BY id DESC LIMIT 10000")
            data = cursor.fetchall()
            for row in data[::-1]:
                state = pickle.loads(zlib.decompress(row[0]))
                continuous_action = pickle.loads(zlib.decompress(row[2]))
                next_state = pickle.loads(zlib.decompress(row[4]))
                gnn_embedding = pickle.loads(zlib.decompress(row[5])) if row[5] else None
                tft_pred = pickle.loads(zlib.decompress(row[6])) if row[6] else None
                multi_tf_data = pickle.loads(zlib.decompress(row[8])) if row[8] else None
                self.buffer.append((state, row[1], continuous_action, row[3], next_state, gnn_embedding, tft_pred, row[7], multi_tf_data))
        logging.info(f"โหลด {len(data)} ข้อมูลจาก SQLite")
        if len(self.buffer) > 50:
            self.fit_anomaly_detector()

    def fit_anomaly_detector(self):
        data = [np.concatenate([e[0].flatten(), np.array([e[1]]), e[2], np.array([e[3]])]) for e in self.buffer]
        self.anomaly_detector.fit(data)
        logging.debug("อัพเดท anomaly detector")

    def __del__(self):
        self.db_conn.close()
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install psutil asyncio numpy

import asyncio
import psutil
import numpy as np
from collections import deque
import logging

class IntelligentResourceManager:
    def __init__(self):
        self.cpu_usage = deque(maxlen=60)
        self.ram_usage = deque(maxlen=60)
        self.model_batch_sizes = {'qnt': 32, 'ddpg': 32, 'ssd': 32, 'evogan': 32, 'tft': 32, 'gnn': 32, 'madrl': 32, 'meta': 32}
        self.task_priorities = {'train': 0.7, 'predict': 0.2, 'data_fetch': 0.1}
        self.resource_lock = asyncio.Lock()

    async def monitor_resources(self):
        process = psutil.Process()
        while CONFIG['system_running']:
            self.cpu_usage.append(process.cpu_percent(interval=1))
            self.ram_usage.append(process.memory_info().rss / (1024 * 1024))
            if CONFIG['resource_adaptive']:
                await self.adjust_resources()
            await asyncio.sleep(60)

    async def adjust_resources(self, trader=None):
        async with self.resource_lock:
            avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 50
            avg_ram = np.mean(self.ram_usage) if self.ram_usage else 1024
            if avg_cpu > (100 - CONFIG['min_cpu_idle_percent']) or avg_ram > (psutil.virtual_memory().total / (1024 * 1024) - CONFIG['min_ram_reserve_mb']):
                for model in self.model_batch_sizes:
                    self.model_batch_sizes[model] = max(1, int(self.model_batch_sizes[model] * 0.8))
                logging.info(f"ลด batch size เนื่องจาก CPU={avg_cpu:.1f}%, RAM={avg_ram:.1f}MB")
            elif avg_cpu < 50 and avg_ram < (psutil.virtual_memory().total / (1024 * 1024) * 0.5):
                for model in self.model_batch_sizes:
                    self.model_batch_sizes[model] = min(128, int(self.model_batch_sizes[model] * 1.2))
                logging.info(f"เพิ่ม batch size เนื่องจาก CPU={avg_cpu:.1f}%, RAM={avg_ram:.1f}MB")
            if trader:
                trader.resource_manager.model_batch_sizes = self.model_batch_sizes
                # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install gym pandas numpy openpyxl gym sqlite3 datetime asyncio keyboard

import gym
import pandas as pd
import numpy as np
import openpyxl
import sqlite3
from datetime import datetime, timedelta
import asyncio
import logging
import keyboard
from collections import deque

class MultiMarketEnv(gym.Env):
    def __init__(self, account_balance=CONFIG['initial_balance'], risk_per_trade=CONFIG['risk_per_trade'], dry_run=CONFIG['dry_run']):
        super().__init__()
        self.account_balance = account_balance
        self.available_balance = account_balance
        self.reinvest_cap = account_balance * 2
        self.initial_balance = CONFIG['initial_balance']
        self.risk_per_trade = risk_per_trade
        self.dry_run = dry_run
        self.symbols = []
        self.positions = {}
        self.current_step = 0
        self.day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.ws_manager = ws_manager  # เชื่อมจาก main
        self.simulator = RealTimeSimulator(self.symbols)  # เชื่อมจาก main
        self.data = {}
        self.raw_data = {}
        self.scalers = {}
        self.trader = None  # เชื่อมจาก main
        self.trade_log_file = CONFIG['trade_log_file']
        self.db_conn = sqlite3.connect('env_history.db', timeout=10)
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS returns (id INTEGER PRIMARY KEY, step INT, return REAL, timestamp REAL)")
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS historical_data (id INTEGER PRIMARY KEY, symbol TEXT, timestamp REAL, close REAL, volume REAL, funding_rate REAL, depth REAL)")
        if not os.path.exists(self.trade_log_file):
            pd.DataFrame(columns=['DateTime', 'Symbol', 'TradeType', 'BuyPrice', 'SellPrice', 'Quantity', 'Capital', 'ProfitLoss']).to_excel(self.trade_log_file, index=False)
        self.min_kpi_threshold = CONFIG['min_daily_kpi']
        self.multi_tf_data = {tf: {} for tf in CONFIG['multi_tf_list']}
        self.kpi_tracker = KPITracker()  # เชื่อมจาก main
        self.kpi_optimizer = KPIOptimizer()  # เชื่อมจาก main
        self.balance_last_updated = 0

    async def load_historical_data(self, symbol, years=CONFIG['historical_years']):
        years_ago = datetime.utcnow() - timedelta(days=365 * years)
        with self.db_conn:
            cursor = self.db_conn.execute("SELECT timestamp, close, volume, funding_rate, depth FROM historical_data WHERE symbol=? AND timestamp>=? ORDER BY timestamp ASC",
                                         (symbol, years_ago.timestamp()))
            data = cursor.fetchall()
            if len(data) < TIMESTEPS and not self.dry_run:
                klines = await exchange.fetch_ohlcv(symbol, timeframe='1h', since=int(years_ago.timestamp() * 1000), limit=17520)
                for kline in klines:
                    timestamp, _, _, _, close, volume = kline
                    self.db_conn.execute("INSERT INTO historical_data (symbol, timestamp, close, volume, funding_rate, depth) VALUES (?, ?, ?, ?, ?, ?)",
                                        (symbol, timestamp / 1000, close, volume, 0.0001, 0))
                self.db_conn.commit()
                cursor = self.db_conn.execute("SELECT timestamp, close, volume, funding_rate, depth FROM historical_data WHERE symbol=? AND timestamp>=? ORDER BY timestamp ASC",
                                             (symbol, years_ago.timestamp()))
                data = cursor.fetchall()
        return pd.DataFrame(data, columns=['timestamp', 'close', 'volume', 'funding_rate', 'depth']) if data else pd.DataFrame()

    async def transfer_from_spot_to_futures(self):
        if self.dry_run:
            shortfall = max(0, CONFIG['initial_balance'] - self.account_balance)
            self.account_balance += shortfall
            self.available_balance += shortfall
            return shortfall
        spot_balance = await exchange.fetch_balance(params={'type': 'spot'})['USDT']['free']
        shortfall = max(0, CONFIG['initial_balance'] - self.account_balance)
        if shortfall > 0 and spot_balance >= shortfall:
            await exchange.transfer('USDT', shortfall, 'spot', 'futures')
            self.account_balance += shortfall
            self.available_balance += shortfall
            logging.info(f"โอน {shortfall:.2f} USDT จาก Spot ไป Futures")
            return shortfall
        else:
            logging.warning(f"ยอด Spot ไม่เพียงพอ: มี {spot_balance:.2f}, ต้องการ {shortfall:.2f}")
            return 0

    async def execute_trade_async(self, symbol, side, size, leverage, stop_loss, take_profit, trailing_stop=None, trailing_take_profit=None):
        current_time = time.time()
        if current_time - self.balance_last_updated > 60:
            self.account_balance = self.ws_manager.get_latest_balance()
            self.available_balance = self.account_balance
            if CONFIG['reinvest_profits']:
                self.available_balance += self.kpi_tracker.total_profit * 0.5
            self.balance_last_updated = current_time

        price = self.ws_manager.get_latest_price(symbol) if not self.dry_run else self.raw_data[symbol]['close'].iloc[-1]
        required_margin = (size * price) / leverage * (1 + TAKER_FEE + SLIPPAGE_DEFAULT)
        if required_margin > self.available_balance:
            logging.warning(f"Margin ไม่พอสำหรับ {symbol}: ต้องการ {required_margin:.2f}, มี {self.available_balance:.2f}")
            return 0

        if not self.trader.risk_guardian.evaluate_position(symbol, price, price, size, leverage, side):
            logging.warning(f"ตำแหน่ง {symbol} ไม่ผ่านการประเมินความเสี่ยง")
            return 0

        if self.dry_run:
            future_step = min(self.current_step + 10, len(self.raw_data[symbol]) - 1)
            future_price = self.raw_data[symbol]['close'].iloc[future_step]
            profit = (future_price - price) * size * leverage * (1 - TAKER_FEE - SLIPPAGE_DEFAULT) * (-1 if side == 'SELL' else 1)
        else:
            await api_manager.set_margin_mode(symbol)
            await api_manager.set_leverage(symbol, leverage)
            callback_rate = CONFIG['trailing_callback_rate'] if trailing_stop else None
            order = await api_manager.create_limit_order_with_trailing(symbol, side, size, price, callback_rate)
            if order:
                profit = 0  # คำนวณจริงจาก close_position
            else:
                return 0

        self.positions[symbol] = {
            'size': size, 'entry': price, 'leverage': leverage, 
            'stop_loss': price * (1 - stop_loss if side == 'BUY' else 1 + stop_loss),
            'take_profit': price * (1 + take_profit if side == 'BUY' else 1 - take_profit), 
            'side': side, 'trailing_stop': trailing_stop, 'trailing_take_profit': trailing_take_profit,
            'highest_price': price if side == 'BUY' else float('inf'),
            'lowest_price': price if side == 'SELL' else float('-inf')
        }
        self.account_balance -= required_margin
        self.available_balance -= required_margin
        self.account_balance += profit
        if CONFIG['reinvest_profits'] and profit > 0:
            reinvest_amount = min(profit * 0.5, self.reinvest_cap - self.available_balance)
            self.available_balance += reinvest_amount
            logging.info(f"Reinvest กำไร {reinvest_amount:.2f} USDT")
        trade_log = pd.DataFrame([{
            'วันที่': datetime.utcnow(), 
            'เหรียญ': symbol, 
            'ประเภทการเทรด': side, 
            'ราคาซื้อ': price if side == 'BUY' else 0, 
            'ราคาขาย': price if side == 'SELL' else 0, 
            'ปริมาณ': size, 
            'ทุน': self.account_balance, 
            'กำไร/ขาดทุน': profit,
            'โหมด': 'จำลอง' if self.dry_run else 'จริง'
        }])
        with pd.ExcelWriter(self.trade_log_file, mode='a', if_sheet_exists='overlay') as writer:
            trade_log.to_excel(writer, index=False, header=False)
        return profit

    async def close_position_async(self, symbol, current_price):
        if self.positions.get(symbol):
            size = self.positions[symbol]['size']
            leverage = self.positions[symbol]['leverage']
            profit = (current_price - self.positions[symbol]['entry']) * size * leverage * (1 - TAKER_FEE) * \
                     (-1 if self.positions[symbol]['side'] == 'SELL' else 1)
            self.account_balance += profit + (size * self.positions[symbol]['entry'] / leverage)
            self.available_balance += (size * self.positions[symbol]['entry'] / leverage)
            del self.positions[symbol]
            return profit
        return 0

    async def process_symbol(self, symbol):
        current_price = self.ws_manager.get_latest_price(symbol) if not self.dry_run else self.raw_data[symbol]['close'].iloc[-1]
        position = self.positions.get(symbol)
        profit = 0
        if position:
            if position['trailing_stop']:
                if position['side'] == 'BUY' and current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    position['stop_loss'] = current_price - position['trailing_stop']
                elif position['side'] == 'SELL' and current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    position['stop_loss'] = current_price + position['trailing_stop']
            if position['trailing_take_profit']:
                if position['side'] == 'BUY' and current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    position['take_profit'] = current_price - position['trailing_take_profit']
                elif position['side'] == 'SELL' and current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    position['take_profit'] = current_price + position['trailing_take_profit']
            if (position['side'] == 'BUY' and (current_price <= position['stop_loss'] or current_price >= position['take_profit'])) or \
               (position['side'] == 'SELL' and (current_price >= position['stop_loss'] or current_price <= position['take_profit'])):
                profit = await self.close_position_async(symbol, current_price)
        reward = profit / self.initial_balance if profit != 0 else 0
        return reward, profit

    async def fetch_multi_tf_data(self, symbol):
        for tf in CONFIG['multi_tf_list']:
            if self.dry_run:
                df = self.simulator.data[symbol]
                if not df.empty:
                    self.multi_tf_data[tf][symbol] = df.resample(tf).last().tail(10)
            else:
                klines = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=10)
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.multi_tf_data[tf][symbol] = df

    async def step(self):
        if self.dry_run:
            self.simulator.simulate_step()
            for symbol in self.symbols:
                state_lstm, _, state_ensemble, _, scaler, raw = self.simulator.get_data(symbol)
                self.data[symbol] = {'lstm': state_lstm, 'ensemble': state_ensemble}
                self.scalers[symbol] = scaler
                self.raw_data[symbol] = raw
        await asyncio.gather(*(self.fetch_multi_tf_data(symbol) for symbol in self.symbols))
        if self.account_balance < CONFIG['initial_balance'] * 0.9:
            await self.transfer_from_spot_to_futures()
        await self.check_new_day()
        rewards = []
        total_profit = 0
        tasks = [self.process_symbol(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Error processing {self.symbols[i]}: {result}")
                rewards.append(0)
                total_profit += 0
            else:
                reward, profit = result
                rewards.append(reward)
                total_profit += profit
        self.current_step += 1
        done = self.current_step >= 1440 or self.account_balance < self.initial_balance * (1 - CONFIG['max_drawdown']) or not CONFIG['system_running']
        with self.db_conn:
            self.db_conn.execute("INSERT INTO returns (step, return, timestamp) VALUES (?, ?, ?)",
                               (self.current_step, total_profit, time.time()))
        if done:
            self.reset()
        return self.get_observation(), rewards, done, {'profit': total_profit}

    async def check_new_day(self):
        now = datetime.utcnow()
        if now >= self.day_start + timedelta(days=1):
            excess = max(0, self.account_balance - self.initial_balance)
            if excess > 0 and not self.dry_run:
                await exchange.fapiPrivate_post_transfer({'asset': 'USDT', 'amount': excess, 'type': 2})
                self.account_balance -= excess
                logging.info(f"โอนกำไรส่วนเกิน {excess:.2f} USDT ไป Spot")
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_observation(self):
        dyn_gen = DynamicIndicatorGenerator(self.trader.evogan, self.trader.gnn, self.multi_tf_data)
        observations = []
        for symbol in self.symbols:
            ind = dyn_gen.generate_synthetic_indicators(symbol)
            obs = []
            for tf in CONFIG['multi_tf_list']:
                tf_ind = ind.get(tf, {
                    'base': {k: 0 for k in ['ATR', 'RSI', 'MACD', 'EMA', 'BB_upper', 'BB_lower', 'SMA', 'Stoch_RSI', 'OBV', 'Volume']},
                    'synthetic': np.zeros(10),
                    'gnn_correlations': np.zeros(5)
                })
                base_values = list(tf_ind['base'].values())
                synthetic_values = list(tf_ind['synthetic'].flatten()[:10])
                gnn_values = list(tf_ind['gnn_correlations'].flatten()[:5])
                obs.extend(base_values + synthetic_values + gnn_values)
            observations.append(np.array(obs))
        return np.array(observations)

    def reset(self):
        self.account_balance = self.initial_balance
        self.available_balance = self.initial_balance
        self.positions = {s: None for s in self.symbols}
        self.current_step = 0
        if self.dry_run:
            self.simulator.reset()
            # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install numpy collections asyncio

import numpy as np
from collections import deque
import asyncio
import logging

class RiskGuardian:
    def __init__(self, max_drawdown=CONFIG['max_drawdown'], cut_loss_threshold=CONFIG['cut_loss_threshold']):
        self.max_drawdown = max_drawdown
        self.cut_loss_threshold = cut_loss_threshold
        self.drawdown_history = deque(maxlen=1440)
        self.positions = {}
        self.total_trades = 0
        self.failed_trades = 0
        self.env = None  # เชื่อมจาก main
        self.dynamic_risk_factor = 1.0
        self.volatility_history = deque(maxlen=60)

    def assess_risk(self, balance, initial_balance):
        current_drawdown = (initial_balance - balance) / initial_balance
        self.drawdown_history.append(current_drawdown)
        if current_drawdown > self.max_drawdown * self.dynamic_risk_factor:
            logging.warning(f"Drawdown เกินขีดจำกัด: {current_drawdown:.2%} > {self.max_drawdown * self.dynamic_risk_factor:.2%}")
            return False
        return True

    def evaluate_position(self, symbol, current_price, entry_price, size, leverage, side):
        unrealized_pnl = (current_price - entry_price) * size * leverage * (1 if side == 'BUY' else -1)
        position_value = size * entry_price / leverage
        loss_ratio = -unrealized_pnl / position_value
        adjusted_threshold = self.cut_loss_threshold * self.dynamic_risk_factor
        if loss_ratio > adjusted_threshold:
            logging.warning(f"ตำแหน่ง {symbol} ขาดทุนเกิน {adjusted_threshold:.2%}: {loss_ratio:.2%}")
            return False
        return True

    async def update_dynamic_risk(self, ws_data):
        volatilities = []
        for symbol in ws_data:
            if 'close' in ws_data[symbol]:
                pct_change = (ws_data[symbol]['close'] - ws_data[symbol].get('prev_close', ws_data[symbol]['close'])) / ws_data[symbol]['close']
                volatilities.append(pct_change)
                ws_data[symbol]['prev_close'] = ws_data[symbol]['close']
        if volatilities:
            avg_volatility = np.std(volatilities)
            self.volatility_history.append(avg_volatility)
            avg_vol_history = np.mean(self.volatility_history) if self.volatility_history else CONFIG['min_volatility_threshold']
            self.dynamic_risk_factor = min(2.0, max(0.5, avg_vol_history / CONFIG['min_volatility_threshold']))
            logging.debug(f"อัพเดท dynamic risk factor: {self.dynamic_risk_factor:.2f}")

    async def emergency_stop(self):
        if not self.env:
            return
        for symbol in list(self.positions.keys()):
            current_price = ws_manager.get_latest_price(symbol)
            await self.env.close_position_async(symbol, current_price)
        logging.critical("หยุดฉุกเฉิน: ปิดทุกตำแหน่ง")
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install numpy

import numpy as np
import logging

class StrategyGenerator:
    def __init__(self, trader, env, risk_guardian):
        self.trader = trader
        self.env = env
        self.risk_guardian = risk_guardian
        self.action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

    async def generate_strategy(self, state, symbol, volatility):
        discrete_pred, continuous_pred = self.trader.predict(state)
        action_idx = np.argmax(discrete_pred[0])
        action = self.action_map[action_idx]
        leverage, size = continuous_pred[0]
        stop_loss = CONFIG['stop_loss_percentage'] * (1 + volatility / CONFIG['min_volatility_threshold'])
        take_profit = stop_loss * 2
        return {
            'action': action,
            'symbol': symbol,
            'size': size,
            'leverage': min(leverage, CONFIG['max_leverage_per_symbol'].get(symbol, 125)),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': stop_loss if volatility > CONFIG['min_volatility_threshold'] else None,
            'trailing_take_profit': take_profit if volatility > CONFIG['min_volatility_threshold'] else None
        }

    async def execute_strategy(self, strategy):
        if strategy['action'] == 'HOLD' or not self.risk_guardian.assess_risk(self.env.account_balance, self.env.initial_balance):
            return 0
        profit = await self.env.execute_trade_async(
            strategy['symbol'], strategy['action'], strategy['size'], strategy['leverage'],
            strategy['stop_loss'], strategy['take_profit'], strategy['trailing_stop'], strategy['trailing_take_profit']
        )
        self.risk_guardian.total_trades += 1
        if profit < 0:
            self.risk_guardian.failed_trades += 1
        return profit
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# ไม่ต้องติดตั้งเพิ่ม (ใช้ built-in)

import logging

class KPIOptimizer:
    def __init__(self):
        self.target_kpi = CONFIG['target_kpi_daily']
        self.min_kpi = CONFIG['min_daily_kpi']

    def optimize(self, current_kpi):
        if current_kpi >= self.target_kpi:
            kpi_factor = min(2.0, current_kpi / self.target_kpi)
        elif current_kpi < self.min_kpi:
            kpi_factor = max(0.5, current_kpi / self.min_kpi)
        else:
            kpi_factor = 1.0
        return kpi_factor
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install numpy asyncio

import numpy as np
import asyncio
import logging

class DynamicRiskAllocator:
    def __init__(self):
        self.base_risk = CONFIG['risk_per_trade']

    async def allocate_risk(self, symbols, ws_data, kpi_factor):
        risk_weights = {}
        for symbol in symbols:
            if symbol in ws_data:
                volatility = np.std([ws_data[symbol]['close']] if 'close' in ws_data[symbol] else [0]) or 0.01
                liquidity = ws_data[symbol].get('volume', 0)
                risk_weights[symbol] = self.base_risk * (1 / (volatility + 1e-6)) * (liquidity / CONFIG['liquidity_threshold']) * kpi_factor
            else:
                risk_weights[symbol] = self.base_risk
        return risk_weights
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install datetime logging

from datetime import datetime, timedelta
import logging

class KPITracker:
    def __init__(self):
        self.total_profit = 0
        self.daily_profit = 0
        self.day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async def update(self, profit):
        self.total_profit += profit
        now = datetime.utcnow()
        if now >= self.day_start + timedelta(days=1):
            self.daily_profit = profit
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            self.daily_profit += profit
        logging.info(f"KPI อัพเดท: กำไรวันนี้={self.daily_profit:.2f}, กำไรรวม={self.total_profit:.2f}")
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install asyncio gc torch

import asyncio
import gc
import torch
import logging

class AutomaticBugFixer:
    def __init__(self):
        self.attempts_left = CONFIG['bug_fix_attempts']

    async def analyze_and_fix(self, error, trader, env):
        if not CONFIG['auto_bug_fix'] or self.attempts_left <= 0:
            return False
        error_str = str(error)
        if "CUDA out of memory" in error_str:
            trader.resource_manager.model_batch_sizes = {k: max(1, v // 2) for k, v in trader.resource_manager.model_batch_sizes.items()}
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("แก้ไข CUDA OOM: ลด batch size และเคลียร์หน่วยความจำ")
        elif "API rate limit" in error_str:
            await asyncio.sleep(60)
            logging.info("แก้ไข API rate limit: รอ 60 วินาที")
        elif "network" in error_str.lower():
            await ws_manager.stop()
            await ws_manager.start(env.symbols)
            logging.info("แก้ไข network error: รีสตาร์ท WebSocket")
        else:
            logging.warning(f"ไม่สามารถแก้ไขบั๊ก: {error_str}")
            self.attempts_left -= 1
            return False
        self.attempts_left -= 1
        return True
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install pandas numpy ta

import pandas as pd
import numpy as np
import ta
import logging

class RealTimeSimulator:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data = {symbol: pd.DataFrame() for symbol in symbols}
        self.step = 0

    def update_symbols(self, symbols):
        self.symbols = symbols
        self.data = {symbol: pd.DataFrame() for symbol in symbols if symbol not in self.data}

    def simulate_step(self):
        for symbol in self.symbols:
            if self.data[symbol].empty:
                df = pd.DataFrame(index=range(1440), columns=['close', 'volume'])
                df['close'] = 10000 * (1 + np.random.normal(CONFIG['sim_trend'], CONFIG['sim_volatility'], 1440))
                df['volume'] = np.random.uniform(50, 500, 1440)
                if np.random.random() < CONFIG['sim_spike_chance']:
                    spike_idx = np.random.randint(0, 1440)
                    df.loc[spike_idx, 'close'] *= 1.1
                self.data[symbol] = df
            else:
                new_price = self.data[symbol]['close'].iloc[-1] * (1 + np.random.normal(CONFIG['sim_trend'], CONFIG['sim_volatility']))
                new_volume = np.random.uniform(50, 500)
                self.data[symbol].loc[len(self.data[symbol])] = [new_price, new_volume]
        self.step += 1

    def get_data(self, symbol):
        df = self.data[symbol]
        if len(df) < 10:
            return np.zeros((10, 7)), None, np.zeros(7), None, None, df
        window = df.tail(10)
        close = window['close'].values
        volume = window['volume'].values
        rsi = ta.momentum.RSIIndicator(close).rsi().values[-1] or 50
        macd = ta.trend.MACD(close).macd().values[-1] or 0
        rv = np.std(close) / np.mean(close) if np.mean(close) != 0 else 0
        funding_rate = 0.0001
        depth = 0
        state_lstm = np.array([close, volume, [rsi]*10, [macd]*10, [rv]*10, [funding_rate]*10, [depth]*10]).T
        state_ensemble = np.array([close[-1], volume[-1], rsi, macd, rv, funding_rate, depth])
        scaler = RobustScaler()
        scaled_lstm = scaler.fit_transform(state_lstm)
        return scaled_lstm, None, state_ensemble, None, scaler, df

    def reset(self):
        self.step = 0
        self.data = {symbol: pd.DataFrame() for symbol in self.symbols}
        # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install ta pandas

import ta
import pandas as pd

class IndicatorCalculator:
    def __init__(self, multi_tf_data):
        self.multi_tf_data = multi_tf_data

    def calculate_indicators(self, symbol):
        indicators = {}
        for tf in CONFIG['multi_tf_list']:
            df = self.multi_tf_data[tf].get(symbol, pd.DataFrame())
            if df.empty:
                continue
            close = df['close'].values
            high = df['high'].values if 'high' in df else close
            low = df['low'].values if 'low' in df else close
            volume = df['volume'].values
            indicators[tf] = {
                'ATR': ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1] if len(close) >= 14 else 0,
                'RSI': ta.momentum.RSIIndicator(close).rsi().iloc[-1] if len(close) >= 14 else 50,
                'MACD': ta.trend.MACD(close).macd().iloc[-1] if len(close) >= 26 else 0,
                'EMA': ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1] if len(close) >= 20 else close[-1],
                'BB_upper': ta.volatility.BollingerBands(close).bollinger_hband().iloc[-1] if len(close) >= 20 else close[-1],
                'BB_lower': ta.volatility.BollingerBands(close).bollinger_lband().iloc[-1] if len(close) >= 20 else close[-1],
                'SMA': ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else close[-1],
                'Stoch_RSI': ta.momentum.StochasticRSIIndicator(close).stochrsi().iloc[-1] if len(close) >= 14 else 0.5,
                'OBV': ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1] if len(close) > 1 else volume[-1],
                'Volume': volume[-1]
            }
        return indicators
    # ติดตั้งไลบรารีที่ต้องใช้สำหรับคลาสนี้ (รันคำสั่งเหล่านี้ใน terminal ก่อนใช้งาน)
# pip install torch numpy torch-geometric

import torch
import numpy as np
from torch_geometric.data import Data

class DynamicIndicatorGenerator:
    def __init__(self, evogan, gnn, multi_tf_data):
        self.evogan = evogan
        self.gnn = gnn
        self.multi_tf_data = multi_tf_data
        self.base_calc = IndicatorCalculator(multi_tf_data)
        self.feature_weights = torch.tensor([
            1.0,  # ATR
            0.5,  # RSI
            1.0,  # MACD
            1.0,  # EMA
            0.5,  # BB_upper
            0.5,  # BB_lower
            0.5,  # SMA
            0.5,  # Stoch_RSI
            1.0,  # OBV
            1.0   # Volume
        ]).to(self.evogan.device)

    def generate_synthetic_indicators(self, symbol):
        synthetic = {}
        base_ind = self.base_calc.calculate_indicators(symbol)
        for tf in CONFIG['multi_tf_list']:
            if tf not in base_ind:
                continue
            base_features = np.array([
                base_ind[tf]['ATR'], base_ind[tf]['MACD'], base_ind[tf]['EMA'],
                base_ind[tf]['OBV'], base_ind[tf]['Volume'],
                base_ind[tf]['RSI'], base_ind[tf]['BB_upper'], base_ind[tf]['BB_lower'],
                base_ind[tf]['SMA'], base_ind[tf]['Stoch_RSI']
            ])
            weighted_features = base_features * self.feature_weights.cpu().numpy()
            synthetic_features = self.evogan.generate(
                torch.FloatTensor(weighted_features).to(self.evogan.device)
            ).cpu().numpy()
            graph = self._create_asset_graph(self.multi_tf_data)
            gnn_features = self.gnn.forward(
                torch.FloatTensor(synthetic_features).to(self.gnn.device), graph
            ).cpu().numpy()
            synthetic[tf] = {
                'base': base_ind[tf],
                'synthetic': synthetic_features,
                'gnn_correlations': gnn_features
            }
        return synthetic

    def _create_asset_graph(self, multi_tf_data):
        num_assets = len(multi_tf_data[CONFIG['multi_tf_list'][0]])
        edges = []
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                edges.append([i, j])
                edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.gnn.device) if edges else torch.tensor([[0, 0]], dtype=torch.long).t().to(self.gnn.device)
        return Data(edge_index=edge_index)
    # ===============================================
# ฟังก์ชัน control_loop: ตรวจสอบการกด 'q' เพื่อหยุดระบบอย่างปลอดภัย
# การใช้งาน: รันใน background เพื่อ monitor keyboard input และหยุดระบบเมื่อกด 'q'
# ===============================================
async def control_loop():
    while GlobalConfig.get('system_running'):
        if keyboard.is_pressed('q'):
            GlobalConfig.set('system_running', False)
            logging.info("ตรวจพบการกด 'q': กำลังหยุดระบบอย่างปลอดภัย...")
            # หยุด WebSocket และปิด exchange
            await ws_manager.stop()
            await exchange.close()
            # หยุด emergency stop ถ้ามีตำแหน่งเปิด
            await risk_guardian.emergency_stop()
            logging.info("ระบบหยุดทำงานและปิดทุกส่วนอย่างปลอดภัย")
            break
        await asyncio.sleep(0.1)
        # ===============================================
# ฟังก์ชัน main: รวมทุกคลาสและรันระบบหลัก
# การใช้งาน: รันระบบทั้งหมด รวมถึงการเทรดจริง/จำลอง, การปรับ leverage, และการซิงค์ข้อมูล
# ===============================================
async def main():
    # โหลด CONFIG จาก GlobalConfig (ถ้ามี override จากไฟล์ภายนอก สามารถเรียก GlobalConfig.load_from_file('path/to/config.json'))
    logging.info("เริ่มต้นระบบการเทรดด้วย CONFIG จาก GlobalConfig")

    # เริ่มต้นคลาสหลักและซิงค์ CONFIG
    api_manager = APIManager()
    ws_manager = WebSocketManager(exchange=api_manager.exchange, time_offset=api_manager.time_offset)
    trader = UnifiedQuantumTrader(GlobalConfig.get('input_dim'))
    risk_guardian = RiskGuardian()
    env = MultiMarketEnv()
    env.trader = trader
    trader.replay_buffer = ReplayBuffer()
    risk_guardian.env = env
    strategy_gen = StrategyGenerator(trader, env, risk_guardian)
    resource_manager = IntelligentResourceManager()
    kpi_optimizer = KPIOptimizer()
    risk_allocator = DynamicRiskAllocator()
    kpi_tracker = KPITracker()
    bug_fixer = AutomaticBugFixer()

    # เริ่มต้น WebSocket และดึงข้อมูล USDT pairs
    await ws_manager.start(['BTC/USDT'])
    await ws_manager.fetch_all_usdt_pairs()
    trader.current_symbols = ws_manager.all_usdt_pairs[:GlobalConfig.get('madrl_agent_count')]
    env.symbols = trader.current_symbols
    env.simulator.update_symbols(env.symbols)
    trader.madrl.update_symbols(env.symbols)

    # รัน monitor resources ใน background
    asyncio.create_task(resource_manager.monitor_resources())

    # โหลดข้อมูลย้อนหลังสำหรับเหรียญที่เลือก
    for symbol in env.symbols:
        await env.load_historical_data(symbol)

    # เริ่มต้น control_loop ใน background
    control_task = asyncio.create_task(control_loop())

    # ตั้งสถานะระบบเป็น running
    GlobalConfig.set('system_running', True)
    step_count = 0

    while GlobalConfig.get('system_running'):
        try:
            # บันทึก checkpoint ตาม interval
            if step_count % GlobalConfig.get('checkpoint_interval') == 0:
                torch.save(trader.qnt.state_dict(), 'qnt_checkpoint.pth')
                logging.info(f"บันทึก checkpoint ที่ step {step_count}")

            # เลือกเหรียญ top coins และอัพเดท symbols
            top_symbols = trader.select_top_coins(ws_manager.all_usdt_pairs, ws_manager.data, kpi_tracker)
            if set(top_symbols) != set(env.symbols):
                env.symbols = top_symbols
                env.simulator.update_symbols(top_symbols)
                trader.madrl.update_symbols(top_symbols)
                await ws_manager.update_symbols(top_symbols)

            # ดำเนินการ step ใน env (รวม fetch multi_tf_data)
            observation, rewards, done, info = await env.step()

            # อัพเดท KPI และปรับ reinvest_cap
            total_profit = info['profit']
            await kpi_tracker.update(total_profit)
            kpi_factor = kpi_optimizer.optimize(kpi_tracker.daily_profit)
            env.reinvest_cap = env.initial_balance * kpi_factor

            # อัพเดท dynamic risk
            await risk_guardian.update_dynamic_risk(ws_manager.data)

            # จัดสรร risk weights
            risk_weights = await risk_allocator.allocate_risk(env.symbols, ws_manager.data, kpi_factor)

            # สร้างและดำเนินการ strategy สำหรับแต่ละ symbol
            for i, symbol in enumerate(env.symbols):
                state = observation[i]
                volatility = np.std([ws_manager.data[symbol]['close']] if symbol in ws_manager.data else [0]) or 0.01
                strategy = await strategy_gen.generate_strategy(state.reshape(1, -1), symbol, volatility)
                strategy['size'] *= risk_weights.get(symbol, GlobalConfig.get('risk_per_trade'))
                profit = await strategy_gen.execute_strategy(strategy)
                multi_tf_data = {tf: env.multi_tf_data[tf].get(symbol, pd.DataFrame()).to_dict()
                                 for tf in GlobalConfig.get('multi_tf_list', [])}
                trader.replay_buffer.add(
                    state,
                    np.argmax(trader.predict(state.reshape(1, -1))[0]),
                    trader.predict(state.reshape(1, -1))[1][0],
                    profit,
                    observation[i],
                    None,
                    None,
                    volatility,
                    multi_tf_data
                )
                await trader.evolve(state.reshape(1, -1), profit, volatility)

            # ฝึกโมเดลตาม interval
            if step_count % GlobalConfig.get('rl_train_interval') == 0 and trader.replay_buffer.buffer:
                batch = trader.replay_buffer.sample(32)
                if batch:
                    states, discrete_actions, continuous_actions, rewards, next_states, _, _, atrs, multi_tf_data = batch
                    await trader.train(states, discrete_actions, continuous_actions, rewards, next_states)
                    await trader.adversarial_train(states)

            # ปรับ model weights ด้วย Bayesian Optimization
            if step_count % GlobalConfig.get('auto_ml_interval') == 0:
                trader.bayes_opt.maximize(init_points=2, n_iter=GlobalConfig.get('bayes_opt_steps'))
                trader.model_weights = trader.bayes_opt.max['params']
                logging.info(f"ปรับน้ำหนักโมเดล: {trader.model_weights}")

            step_count += 1
            await asyncio.sleep(60)

        except Exception as e:
            if await bug_fixer.analyze_and_fix(e, trader, env):
                logging.info("แก้ไขบั๊กสำเร็จ ทำงานต่อ")
                continue
            logging.critical(f"ข้อผิดพลาดร้ายแรง: {e}")
            traceback.print_exc()
            await risk_guardian.emergency_stop()
            break

    # ยกเลิก control_task เมื่อหยุดระบบ
    control_task.cancel()


if __name__ == "__main__":
    auto_install_all()
    asyncio.run(main())
