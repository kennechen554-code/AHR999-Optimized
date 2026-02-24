```
██╗  ██╗███████╗███╗   ██╗███╗   ██╗███████╗    ██╗███╗   ██╗██████╗ ███████╗██╗  ██╗
██║ ██╔╝██╔════╝████╗  ██║████╗  ██║██╔════╝    ██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝
█████╔╝ █████╗  ██╔██╗ ██║██╔██╗ ██║█████╗      ██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝ 
██╔═██╗ ██╔══╝  ██║╚██╗██║██║╚██╗██║██╔══╝      ██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗ 
██║  ██╗███████╗██║ ╚████║██║ ╚████║███████╗    ██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝    ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

<div align="center">

**智能定投算法 · 多维市场洞察 · 全自动执行**

*Quantitative DCA · Multi-Signal Intelligence · Zero-Cost Automation*

---

`Python 3.10+` &nbsp;|&nbsp; `FastAPI` &nbsp;|&nbsp; `ccxt` &nbsp;|&nbsp; `8 Exchanges` &nbsp;|&nbsp; `GitHub Actions`

</div>

---

> ⚠️ **免责声明**：本项目仅供学习研究使用，不构成任何投资建议。加密货币具有极高波动性，请在充分理解模型原理与风险后自行决策。使用前请参阅 `kenne_backtest.html` 查看历史回测数据。

---

## 目录

- [核心理念](#核心理念)
- [指标深度解析](#指标深度解析)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [Web 应用部署](#web-应用部署)
- [GitHub Actions 部署](#github-actions-部署)
- [功能详解](#功能详解)
- [配置参考](#配置参考)
- [文件结构](#文件结构)
- [常见问题](#常见问题)

---

## 核心理念

**Kenne Index** 是对传统 AHR999 的改进与延伸，在估值计算的基础上内置了动量过滤，并提供多个辅助指标供交叉验证。

传统 AHR999 只告诉你价格是否低估，却无法判断当前是否处于自由落体的刀口之上。本系统分为两层：

```
── 核心定投指标 ──────────────────────────────────────────
  Kenne Index       幂律增长估值 × 200日几何均线偏离度
                    内置防落刀 Momentum Filter
                    → 直接驱动买入倍数与资金分配

── 辅助参考指标（不影响定投逻辑，仅供判断市场状态）──────
  MVRV-Z Score      链上成本与当前市值的偏离程度
  盘口深度           OKX 永续合约买卖一档及资金费率
```

---

## 指标深度解析

### Kenne Index

基于 AHR999 原始公式，融合幂律增长模型，并引入**自动斜率重拟合**机制：

```
Kenne Index = (当前价格 / 200日几何均线) × (当前价格 / 幂律增长估值)
幂律估值    = 10 ^ (slope × log10(距创世天数) + intercept)
```

每次运行时，系统自动用本地全量历史日线回归重拟合 `slope` 和 `intercept`，
结果持久化至 `model_params.json`，随 CSV 一起提交回仓库，可追踪参数漂移。

**信号区间：**

| Kenne Index | 市场含义 | 默认执行倍数 |
|-------------|---------|------------|
| `< 0.45` | 历史极低估区间 | **2.0x 重仓** |
| `0.45 ~ 1.20` | 合理定投价值区间 | **1.0x 正常** |
| `> 1.20` | 高估，风险回避 | **0x 停止** |

> SOL 上沿阈值为 1.50，因历史数据较短，分布中位数更高。

**各资产幂律参数（当前拟合值）：**

| 资产 | 斜率 | 创世日期 | R² | 数据年限 |
|------|------|---------|-----|---------|
| BTC | 4.7733 | 2009-01-03 | 0.78 | 15+ 年 |
| ETH | 1.9849 | 2015-07-30 | 0.58 | 10 年 |
| SOL | 1.4426 | 2020-03-16 | 0.53 | 5.5 年 |

---

### MVRV-Z Score（辅助指标）

Market Value to Realized Value Z-Score —— 衡量市场整体浮盈程度的链上指标，作为 Kenne Index 信号的**辅助参考**，不直接参与定投倍数计算。

系统通过本地 CSV 数据 + CoinGecko 接口在本地计算近似 MVRV-Z：

```
已实现价格  = 1100 日滚动 VWAP（代理 Realized Price）
已实现市值  = 已实现价格 × 流通供应量
标准化因子  = 全历史价格标准差 × 流通供应量
MVRV-Z     = (当前市值 - 已实现市值) / 标准化因子
```

**参考阈值：**

| MVRV-Z | 解读 |
|--------|------|
| `< 0` | 深度低估，历史抄底区 |
| `0 ~ 0.5` | 正常波动范围 |
| `> 0.5` | 市场过热，谨慎追高 |

> 对齐参考基准：Coinglass BTC Z-Score

---

### 防落刀过滤器（Momentum Filter）

价格低估 ≠ 安全买入。系统实时检测近期价格动量：

| 状态 | 判定条件 | 执行倍数 |
|------|---------|---------|
| `STABLE` | 7日跌幅在正常范围内 | **1.0x**（足额执行）|
| `STABILIZING` | 急跌后已从低点反弹 ≥ 5% | **0.75x**（择机参与）|
| `FALLING` | 7日跌 > 15% 且无有效反弹 | **0.40x**（轻仓防御）|

> FALLING 保留 0.4x 而非 0：历史底部往往诞生于持续下跌中，完全缺席会错失关键入场点。

---

### 实时市场数据（需 Redis）

后台每 **15 分钟**自动刷新：

- **买卖盘口深度**：OKX V5 永续合约一档 Bid/Ask 价格与数量
- **永续合约资金费率**：正值 = 多头溢价，负值 = 空头溢价
- **MVRV 缓存**：CoinGecko 市值数据缓存 1 小时，避免频繁请求

后台每 **4 小时**自动拉取最新 K 线，追加到本地 CSV。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                  Kenne Index Web App                     │
│               http://localhost:8000                      │
└───────────────────────┬──────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
  ┌─────────────┐ ┌──────────┐ ┌───────────┐
  │kenne_index  │ │config    │ │dca_log    │
  │.py          │ │.json     │ │.json      │
  │• Refit      │ └──────────┘ └───────────┘
  │• Kenne Index│
  │• Momentum   │      ┌───────────────────────┐
  └─────────────┘      │    Background Tasks   │
          │            │  ┌─────────────────┐  │
          ▼            │  │ K-line Update   │  │
  ┌─────────────┐      │  │   every 4h      │  │
  │ *.csv       │◄─────│  ├─────────────────┤  │
  │ (local DB)  │      │  │ Depth + Funding │  │
  └─────────────┘      │  │   every 15min   │  │
                       │  └─────────────────┘  │
                       └───────────────────────┘
                                  │
                       ┌──────────┴──────────┐
                       │  Redis (Optional)   │
                       │ depth/funding/cache │
                       └─────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
          ┌──────────────┐ ┌──────────┐ ┌──────────┐
          │ ccxt · 8 Ex  │ │CoinGecko │ │   SMTP   │
          │ OKX Binance  │ │   API    │ │  Email   │
          │ Bybit Bitget │ └──────────┘ └──────────┘
          │ Gate KuCoin  │
          │ HTX  MEXC    │
          └──────────────┘
```

---

## 快速开始

### 环境要求

- **Python 3.10+**
- **Redis**（可选 —— 无 Redis 时实时深度/资金费率/MVRV 显示 N/A，定投功能完全正常）

### 安装依赖

```bash
pip install fastapi "uvicorn[standard]" ccxt pandas numpy scipy requests redis
```

---

## Web 应用部署

### 第一步：准备文件

```
项目目录/
├── main.py
├── kenne_index.py
├── kenne_dca.py
├── templates/
│   └── index.html
├── btc_4h_data_2018_to_2025.csv
├── eth_4h_data_2017_to_2025.csv
└── sol_4h_data_2020_to_2025.csv
```

### 第二步：启动

```bash
uvicorn main:app --reload
# 访问 http://127.0.0.1:8000
```

### 第三步：配置

进入 **CONFIG** 页面完成以下配置：

#### 交易所 API

系统支持 8 家交易所，通过 ccxt 统一接入：

| 交易所 | 需要 Passphrase？ | API 申请建议 |
|--------|-----------------|-------------|
| OKX | ✅ | 仅勾选「交易」权限，不勾选「提币」 |
| Binance | ❌ | 勾选「现货交易」权限 |
| Bybit | ❌ | 勾选「现货」权限 |
| Bitget | ✅ | 勾选「现货交易」权限 |
| Gate.io | ❌ | 勾选「现货」权限 |
| KuCoin | ✅ | Passphrase = Trading Password |
| HTX | ❌ | 标准 API Key + Secret |
| MEXC | ❌ | 标准 API Key + Secret |

> 选择交易所后，界面会自动显示该交易所的 API 申请说明。

#### 邮件 SMTP

| 邮箱 | Host | Port | 密码类型 |
|------|------|------|---------|
| Gmail | `smtp.gmail.com` | `587` | 应用专用密码（需开启两步验证）|
| QQ | `smtp.qq.com` | `465` | 邮箱授权码 |
| 163 | `smtp.163.com` | `465` | 邮箱授权码 |
| Outlook | `smtp-mail.outlook.com` | `587` | 账户密码 |

#### 预算策略

| 模式 | 说明 | 示例 |
|------|------|------|
| `MONTHLY` | 月度上限自动均摊，防止超支 | 月预算 $700，周投 ≈ $163/次 |
| `FIXED` | 每次执行固定金额，无月度追踪 | 每次 $175，自行控制总量 |

### 第四步：验证信号

**DASHBOARD → REFRESH** → 查看三枚信号卡片是否正常加载。

### 第五步：测试执行

**EXECUTE → DRY RUN** → 确认分配逻辑无误 → 关闭模拟盘 → **EXECUTE LIVE**

### 第六步：（可选）启用 Redis

```bash
# macOS
brew install redis && brew services start redis

# Ubuntu / Debian
sudo apt install redis-server && sudo systemctl start redis

# Docker（推荐）
docker run -d -p 6379:6379 redis:alpine
```

Redis 就绪后重启 uvicorn，实时深度与资金费率自动激活。

---

## GitHub Actions 部署

无需服务器，每周自动触发，发送邮件信号，更新的 K 线数据自动提交回仓库。

### 第一步：Fork 仓库

右上角 **Fork** → 选择你的账号 → **Create fork**

> 建议设为 **Private（私有仓库）**，防止配置信息被意外公开。

### 第二步：添加 Secrets

**Settings → Secrets and variables → Actions → New repository secret**

| Secret | 内容 | 必填 |
|--------|------|------|
| `SMTP_HOST` | 邮件服务器地址 | ✅ |
| `SMTP_PORT` | 端口（587 / 465）| ✅ |
| `SMTP_USER` | 发件邮箱 | ✅ |
| `SMTP_PASSWORD` | 邮箱授权码 | ✅ |
| `EMAIL_TO` | 收件邮箱 | ✅ |
| `OKX_API_KEY` | OKX API Key | 仅自动交易时 |
| `OKX_API_SECRET` | OKX Secret | 仅自动交易时 |
| `OKX_API_PASSPHRASE` | OKX Passphrase | 仅自动交易时 |

### 第三步：配置策略

编辑 `.github/workflows/weekly_notify.yml`：

```yaml
on:
  schedule:
    - cron: '0 22 * * 0'   # 周日 22:00 UTC = 周一 06:00 CST（推荐，错开高峰）
  workflow_dispatch:         # 支持手动触发

env:
  BUDGET_MODE:        MONTHLY   # MONTHLY = 月度均摊 | FIXED = 每次固定
  BUDGET_AMOUNT:      '700'     # MONTHLY 月上限 / FIXED 单次金额（USDT）
  RUN_INTERVAL_DAYS:  '7'       # 必须与 cron 周期一致
  SIMULATED:          'true'    # true=模拟盘  false=真实交易
```

**cron 与 `RUN_INTERVAL_DAYS` 对照：**

| 频率 | cron | RUN_INTERVAL_DAYS |
|------|------|------------------|
| 每周一 06:00 CST | `0 22 * * 0` | `7` |
| 每天 09:00 CST | `0 1 * * *` | `1` |
| 每月 1 日 | `0 1 1 * *` | `30` |

### 第四步：手动测试

**Actions → AHR999 Auto Invest → Run workflow**

✅ 绿色运行 + 收到邮件 = 部署成功，之后按 cron 自动执行。

---

## 功能详解

### Web 界面

#### DASHBOARD

- 三枚 Kenne Index 信号卡片（颜色随区间变化：红=极低估 / 蓝=定投区 / 灰=观望）
- 本次分配预览（权重进度条 + USDT 金额实时计算）
- 幂律模型参数表（slope / R² / 数据年限，R² < 0.65 显示 ⚠ 警告）
- MVRV-Z Score 辅助面板（链上浮盈参考，不影响定投逻辑）及实时盘口深度

#### CONFIG

- 交易所下拉选择，自动显示/隐藏 Passphrase 字段及申请说明
- 模拟盘 / 真实盘一键切换
- MONTHLY / FIXED 预算模式切换，实时显示每次预估金额
- SMTP 常用邮箱预设 + 自定义 Host 选项

#### EXECUTE

| 操作 | 说明 |
|------|------|
| **FETCH CANDLES** | 从交易所拉取最新 4H K 线，追加到本地 CSV |
| **CHECK BALANCE** | 查询账户 USDT 及持仓余额 |
| **DRY RUN** | 模拟完整定投流程，记录日志，不产生真实订单 |
| **EXECUTE LIVE** | 真实定投（需关闭模拟盘 + 有效 API Key）|
| **SEND EMAIL** | 计算当前信号并发送邮件报告，不交易 |

所有操作实时输出日志：✓ 绿色 = 成功，✗ 红色 = 失败。

#### HISTORY

- 按月筛选历史记录
- 每笔显示：时间 / 币种 / 交易所 / 金额 / Kenne 值 / 动量 / 状态
- 汇总本期总投入金额与记录条数

### CLI 模式（无 Web，适合 GitHub Actions / crontab）

```bash
python3 kenne_dca.py --notify          # 计算信号 + 发送邮件，不交易
python3 kenne_dca.py --dry-run         # 模拟完整流程，记录日志
python3 kenne_dca.py                   # 真实执行一次定投
python3 kenne_dca.py --daemon          # 守护进程（按 RUN_INTERVAL_DAYS 循环）
python3 kenne_dca.py --history         # 查看全部历史
python3 kenne_dca.py --history 2026-03 # 查看指定月份

# crontab 示例（每周一 09:00 发邮件）
0 9 * * 1 cd /path/to/project && python3 kenne_dca.py --notify >> kenne.log 2>&1
```

---

## 配置参考

### 资金分配算法

```
每币 final_mult  = base_mult（Kenne 区间倍数）× momentum_mult（动量倍数）
每币权重        = final_mult / sum(所有活跃币种 final_mult)

单币权重上限：BTC ≤ 60%  /  ETH ≤ 50%  /  SOL ≤ 50%
（3 轮迭代归一化，防止极端行情下权重失控）
```

### MONTHLY 预算计算逻辑

```
每月执行次数   = 30 / RUN_INTERVAL_DAYS
目标单次金额  = BUDGET_AMOUNT / 每月执行次数
本月剩余额度  = BUDGET_AMOUNT - 本月已消费（读取 dca_log.json）
本月剩余次数  = 本月剩余天数 / RUN_INTERVAL_DAYS
本次金额      = min(目标单次, 本月剩余额度 / 本月剩余次数)
```

月度耗尽后自动跳过，邮件显示「本次停止定投」，不会超支。

### 环境变量完整列表

```bash
# 交易所 API
OKX_API_KEY=xxx
OKX_API_SECRET=xxx
OKX_API_PASSPHRASE=xxx

# 邮件
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=app_password
EMAIL_TO=your@gmail.com

# 定投策略
BUDGET_MODE=MONTHLY       # MONTHLY / FIXED
BUDGET_AMOUNT=700         # USDT
RUN_INTERVAL_DAYS=7       # 天数

# 模式
SIMULATED=true            # true / false

# Redis（可选）
REDIS_URL=redis://localhost:6379/0

# CoinGecko（可选，提升速率限制）
COINGECKO_API_KEY=CG-xxxx
```

---

## 文件结构

```
项目根目录/
│
├── main.py                       # FastAPI 后端核心
├── kenne_index.py                # Kenne Index 信号引擎（含自动重拟合）
├── kenne_dca.py                  # CLI 定投执行脚本
│
├── templates/
│   └── index.html                # Web 前端（单文件，内嵌 CSS + JS）
│
├── config.json                   # 本地配置（⚠ 含 API Key，勿提交）
├── dca_log.json                  # 交易历史（自动生成）
├── model_params.json             # 重拟合参数缓存（自动生成 + 自动提交）
│
├── btc_4h_data_2018_to_2025.csv  # BTC 历史 K 线（每次运行自动追加）
├── eth_4h_data_2017_to_2025.csv  # ETH 历史 K 线
├── sol_4h_data_2020_to_2025.csv  # SOL 历史 K 线
│
├── kenne_backtest.html           # 历史回测报告（使用前请先阅读）
├── requirements.txt              # Python 依赖
├── _verify_kenne.py              # 参数验证脚本（开发用）
│
└── .github/workflows/
    └── weekly_notify.yml         # GitHub Actions 工作流
```

---

## 常见问题

**Q：启动报错 `Could not import module "main"`？**

确认命令在包含 `main.py` 的目录执行，且依赖完整：

```bash
pip install fastapi "uvicorn[standard]" ccxt pandas numpy scipy requests redis
```

---

**Q：没有 Redis，功能受限吗？**

不影响核心功能。Redis 是可选组件，系统启动时自动检测：

| 功能 | 有 Redis | 无 Redis |
|------|---------|---------|
| Kenne Index 信号 | ✅ | ✅ |
| 预算管理 / 定投 | ✅ | ✅ |
| 邮件通知 | ✅ | ✅ |
| 实时盘口深度 | ✅ | N/A |
| 永续资金费率 | ✅ | N/A |
| MVRV-Z Score | ✅（缓存） | ✅（实时请求）|

---

**Q：GitHub Actions 没有准时触发？**

GitHub 免费 tier 在周一高峰期延迟 1~5 小时属正常现象。建议改为周日深夜触发，错开高峰：

```yaml
- cron: '0 22 * * 0'   # 周日 22:00 UTC = 周一 06:00 CST
```

---

**Q：MVRV-Z 与 Coinglass 数据有偏差？**

本地 MVRV-Z 是基于 1100 日 VWAP 的近似值，参考趋势方向，不作精确信号使用。Coinglass 使用完整 UTXO 链上数据，更为精确。

---

**Q：幂律参数（slope）会自动更新吗？**

会。每次计算信号时，系统用全量历史日线自动回归，新参数写入 `model_params.json`，GitHub Actions 运行后自动提交回仓库。可在 Web 界面 DASHBOARD 的参数表或直接查看该文件追踪变化。

---

**Q：如何只收邮件、手动定投？**

`weekly_notify.yml` 默认即为此模式（使用 `--notify` 命令），无需 OKX API Secrets。  
每周收到邮件 → 查看建议金额 → 手动登录交易所下单。

---

## 致谢

- **ahr999** — AHR999 指标创始人
- **PlanB** — 比特币 S2F 与幂律模型研究者
- **CCXT** — 开源多交易所统一 API 库

---

<div align="center">

**本项目仅供学习研究，不构成投资建议。市场有风险，投资需谨慎。**

*Made for the DCA community · MIT License*

</div>
