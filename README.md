# AHR999 Optimized自动化定投系统 (Open Source)

基于 AHR999 指标和幂律法则的自动化加密货币定投工具。支持 OKX 交易所，可部署在 GitHub Actions 或本地服务器。

程序会自动计算 BTC / ETH / SOL 的抄底信号，根据价格动量（急跌/企稳）动态调整购买权重，并自动执行下单或发送邮件提示。

## 主要特性

* **智能定投**：结合 AHR999 指标与幂律估值，自动判断抄底、定投或观望。
* **动态策略**：支持“固定金额”和“月度预算”两种模式。
* **完全自定义**：支持按 **日 / 周 / 月** 频率运行。
* **数据自持**：K 线数据保存在仓库 CSV 中，每次运行自动补全，不依赖外部付费 API。

## 部署步骤

### 1. Fork 仓库
将本项目 Fork 到你的 GitHub 账号。

### 2. 设置 Secrets
进入仓库 Settings -> Secrets and variables -> Actions -> New repository secret，填入以下敏感信息：

| Secret 名称      | 说明                          |
|-----------------|-------------------------------|
| `OKX_API_KEY`     | OKX API Key (申请时勾选交易权限) |
| `OKX_API_SECRET`  | OKX API Secret                |
| `OKX_API_PASSPHRASE` | OKX Passphrase             |
| `SMTP_HOST`       | 邮件服务器 (如 smtp.gmail.com) |
| `SMTP_PORT`       | 端口 (587 或 465)             |
| `SMTP_USER`       | 发件邮箱地址                   |
| `SMTP_PASSWORD`   | 邮箱授权码/应用专用密码         |
| `EMAIL_TO`        | 接收定投报告的邮箱             |

### 3. 配置定投策略
修改 `.github/workflows/auto_invest.yml` 文件中的 `env` 部分，或在仓库 Variables 中设置：

| 变量名 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| `BUDGET_MODE` | `MONTHLY` | `MONTHLY`: 按月预算自动分配 (自动计算剩余额度)<br>`FIXED`: 每次运行投入固定金额 |
| `BUDGET_AMOUNT` | `700` | 月度总预算 或 单次固定金额 (USDT) |
| `RUN_INTERVAL_DAYS` | `7` | 运行间隔天数 (用于计算月度剩余次数)。<br>每周填 `7`，每天填 `1` |

### 4. 设置运行频率
修改 `.github/workflows/auto_invest.yml` 中的 `cron` 表达式以调整定投频率：

* **每周一 09:00 CST (默认)**: `0 1 * * 1`
* **每天 09:00 CST**: `0 1 * * *`
* **每月 1 日 09:00 CST**: `0 1 1 * *`

### 5. 启动
手动触发一次：Actions -> AHR999 Auto Invest -> Run workflow。
检查邮箱是否收到信号报告，确认配置无误。

## 本地运行 / 守护进程

若在本地服务器或 Docker 中运行，可使用以下命令：

```bash
# 安装依赖
pip install pandas numpy scipy requests

# 1. 仅发送信号邮件 (不交易)
python3 ahr999_dca.py --notify

# 2. 真实交易 (需设置环境变量)
export OKX_API_KEY="xxx" ... (其他密钥)
python3 ahr999_dca.py

# 3. 守护进程模式 (长期运行)
# 配合 RUN_INTERVAL_DAYS 自动休眠
python3 ahr999_dca.py --daemon
