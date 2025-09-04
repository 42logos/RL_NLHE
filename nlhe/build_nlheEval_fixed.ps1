param(
  # 你的 Python 虚拟环境路径（默认 .\.venv）
  [string]$VenvPath = ".\.venv",
  # Rust crate 目录（包含 Cargo.toml，默认 nlhe_eval）
  [string]$CrateDir = "nlhe_eval",
  # 使用 maturin 构建并直接安装到 venv（可选）
  [switch]$UseMaturin
)

$ErrorActionPreference = "Stop"

function Fail([string]$msg) {
  Write-Host "错误: $msg" -ForegroundColor Red
  exit 1
}

function Ensure-Cmd([string]$name, [string]$hint) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    Fail "未找到命令：$name。$hint"
  }
}

Write-Host "=== 构建 nlhe_eval Python 扩展（Windows）===" -ForegroundColor Cyan

# 进入 crate 目录
if (-not (Test-Path $CrateDir)) { Fail "找不到目录：$CrateDir" }
Push-Location $CrateDir

# 定位 venv Python
$venvPython = Join-Path (Resolve-Path (Join-Path ".." $VenvPath)) "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  # 兜底：用 py 启动器
  if (Get-Command py -ErrorAction SilentlyContinue) {
    $venvPython = "py -3"
    Write-Host "未找到 $VenvPath 的 python，改用系统 Python（py -3）" -ForegroundColor Yellow
  } else {
    Pop-Location
    Fail "找不到虚拟环境 Python：$VenvPath\Scripts\python.exe（且无系统 Python 启动器 py）"
  }
}

# 基本架构一致性检查
try {
  $pyBits = & $venvPython -c "import struct; print(struct.calcsize('P')*8)"
  if (Get-Command rustc -ErrorAction SilentlyContinue) {
    $rustHost = (rustc -vV) | Select-String "host: "
    if ($rustHost) {
      $host = $rustHost.ToString().Split()[-1]
      if ($pyBits -eq "64" -and ($host -notmatch "x86_64")) {
        Write-Host "警告：Python 为 64 位，但 rustc host=$host 不是 x86_64，可能导致导入失败。" -ForegroundColor Yellow
      }
    }
  }
} catch { }

if ($UseMaturin) {
  # ----------- maturin 路线：最省心 -----------
  Ensure-Cmd "cargo" "请安装 Rust（rustup）并把 cargo 加入 PATH。"
  if (-not (Get-Command maturin -ErrorAction SilentlyContinue)) {
    Write-Host "未检测到 maturin，开始在 venv 中安装..." -ForegroundColor Yellow
    & $venvPython -c "import sys,subprocess; subprocess.check_call([sys.executable,'-m','pip','install','maturin'])"
  }
  Write-Host "使用 maturin develop --release 安装到虚拟环境..." -ForegroundColor Cyan
  & $venvPython -m maturin develop --release -m (Join-Path (Get-Location) "Cargo.toml")
  Pop-Location

  Write-Host "测试模块导入..." -ForegroundColor Cyan
  & $venvPython -c @"
import nlhe_eval, sys
print("✓ 模块导入成功")
print("可用成员（不含私有）:", [f for f in dir(nlhe_eval) if not f.startswith("_")])
"@
  Write-Host "`n✓ 完成（maturin 路线）" -ForegroundColor Green
  exit 0
}

# ----------- 纯 cargo 路线 -----------
Ensure-Cmd "cargo" "请安装 Rust（rustup）并把 cargo 加入 PATH。"

Write-Host "使用 cargo 构建 release..." -ForegroundColor Cyan
cargo build --release

$releaseDir = Join-Path "target" "release"

# 优先查找 *.pyd（pyo3/成熟打包一般会有）
$pyd = @(Get-ChildItem -Path $releaseDir -Filter "*nlhe_eval*.pyd" -File -ErrorAction SilentlyContinue)
if (-not $pyd) {
  # 也搜 deps 目录
  $depsDir = Join-Path $releaseDir "deps"
  if (Test-Path $depsDir) {
    $pyd = @(Get-ChildItem -Path $depsDir -Filter "*nlhe_eval*.pyd" -File -ErrorAction SilentlyContinue)
  }
}

# 若没有 .pyd，尝试 .dll 并重命名为 nlhe_eval.pyd（仅当你的 crate 是 pyo3 扩展模块时有效）
$dll = $null
if (-not $pyd -or $pyd.Count -eq 0) {
  $dll = @(Get-ChildItem -Path $releaseDir -Filter "*nlhe_eval*.dll" -File -ErrorAction SilentlyContinue)
  if (-not $dll -or $dll.Count -eq 0) {
    $depsDir = Join-Path $releaseDir "deps"
    if (Test-Path $depsDir) {
      $dll = @(Get-ChildItem -Path $depsDir -Filter "*nlhe_eval*.dll" -File -ErrorAction SilentlyContinue)
    }
  }
  if (-not $dll -or $dll.Count -eq 0) {
    Pop-Location
    Fail "未找到构建产物（*.pyd 或 *.dll）于 $releaseDir"
  }
}

$artifactOut = Join-Path $releaseDir "nlhe_eval.pyd"
if ($pyd -and $pyd.Count -gt 0) {
  $src = ($pyd | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
  Copy-Item $src $artifactOut -Force
} else {
  $src = ($dll | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
  Copy-Item $src $artifactOut -Force
}

if (-not (Test-Path $artifactOut)) {
  Pop-Location
  Fail "未能生成 $artifactOut"
}

Write-Host "✓ 构建产物: $artifactOut" -ForegroundColor Green

# 解析 site-packages（更稳：优先 sysconfig）
$sitePackages = & $venvPython -c @"
import sysconfig, site
print(sysconfig.get_paths().get('platlib') or sysconfig.get_paths()['purelib'])
"@
$sitePackages = $sitePackages.Trim()
if (-not $sitePackages) {
  Pop-Location
  Fail "无法获取 site-packages 路径"
}
Write-Host "Python site-packages: $sitePackages" -ForegroundColor Yellow

# 复制到 venv
$dest = Join-Path $sitePackages "nlhe_eval.pyd"
Copy-Item $artifactOut $dest -Force
Write-Host "✓ 已复制到: $dest" -ForegroundColor Green

Pop-Location

# 导入自检
Write-Host "测试模块导入..." -ForegroundColor Cyan
& $venvPython -c @"
import nlhe_eval, sys
print("✓ 模块导入成功")
print("可用成员（不含私有）:", [f for f in dir(nlhe_eval) if not f.startswith("_")])
"@

Write-Host "`n✓ 完成（cargo 路线）" -ForegroundColor Green
Write-Host "提示：如导入失败，请确认 Cargo.toml 中 [lib] 使用 cdylib 且启用 pyo3 扩展" -ForegroundColor DarkYellow
