# 第三节 使用 Docker Compose 部署模型服务

尽管通过 `uv` 高效管理环境，并借助 `Gunicorn` 和 `systemd` 的方式能够成功部署 NER 模型 API，但在团队协作和多环境部署的场景下，它仍可能面临“在我机器上明明是好的”这类环境不一致的挑战。此外，每次部署新项目或迁移服务器时，重复进行环境配置也显得效率不高。

为了追求更高层次的部署自动化与环境一致性，我们将引入容器化技术——**Docker**。本节将学习如何将 NER 应用打包成一个 Docker 镜像，并使用 **Docker Compose** 来定义和运行它。通过这种方式，我们可以将应用及其所有依赖（包括 Python 解释器本身）封装在一个隔离的、标准化的“集装箱”里，实现真正的“一次构建，随处运行”。

## 一、Docker 与 Docker Compose

### 1.1 什么是 Docker？

传统的服务器部署方式，即便省去了安装操作系统的步骤，也依然需要在服务器上手动配置运行环境、安装项目依赖库。这个过程不仅繁琐，而且在不同机器上难以保证环境的完全一致，容易出错。

**Docker** 改变了这一模式。我们可以把它想象成一个软件“集装箱”技术。它允许开发者将应用及其所有依赖项（代码、运行时、系统工具、库）打包到一个轻量级、可移植的**容器**中。这个容器可以在任何安装了 Docker 的机器上运行，无论是开发者的笔记本电脑、测试服务器还是生产环境的云服务器，都能保证环境的完全一致。

Docker 的核心概念包括：

- **镜像 (Image)**: 一个只读的模板，用于创建容器。它像一个“安装包”，其中包含了运行应用所需的一切，如代码、一个迷你的操作系统以及所有依赖库。以我们的 NER 项目为例，可以基于一个包含 Python 3.10 的官方镜像，再打包进相关的代码和依赖。
- **容器 (Container)**: 镜像的运行实例。容器是轻量级的，因为它与宿主机共享操作系统内核，启动速度极快，资源占用也远小于传统虚拟机。每个容器都运行在自己独立、隔离的环境中。
- **Dockerfile**: 一个文本文档，包含了一系列指令，用于告诉 Docker 如何自动构建一个镜像。例如，从哪个基础镜像开始，需要安装哪些软件，复制哪些文件，以及容器启动时要执行什么命令。

### 1.2 为何选择 Docker Compose？

当我们使用 Docker 时，会用到 `docker run`、`docker build` 等命令行工具来构建镜像和运行容器。对于单个容器的应用，这些命令尚可应付。但当应用变得复杂，比如需要同时运行一个 API 服务、一个数据库和一个缓存服务时，手动管理多个容器的启动顺序、网络连接和数据卷会变得异常繁琐。

**Docker Compose** 就是 Docker 官方提供的解决方案。它是一个用于定义和运行多容器 Docker 应用的工具。通过一个名为 `docker-compose.yml` 的 YAML 文件，可以配置应用所需的所有服务。然后，只需一个简单的命令 `docker compose up`，就能够根据配置文件创建并启动所有服务。

对于我们当前的单个 NER API 服务，使用 Docker Compose 也能带来诸多好处：
- **配置集中化**: 将所有部署相关的参数（如构建指令、端口映射、重启策略）都写在一个文件中，一目了然。
- **命令简化**: 将冗长的 `docker run` 命令简化为 `docker compose up`，更易于记忆和使用。
- **易于扩展**: 将来若需增加数据库等新服务，仅需在配置文件中添加几行即可。

## 二、环境准备

### 2.1 云服务器

服务器的准备工作，包括创建实例、配置安全组放行 **8000 端口**等，与前文介绍的步骤相同。确保我们已拥有一台可以通过 SSH 连接的、配置好安全组的云服务器。

### 2.2 安装 Docker 和 Docker Compose

登录到云服务器后，我们需要安装 Docker 环境。在安装之前，同样需要使用 `sudo apt update` 更新一下系统的包索引，保证获取到的是最新的软件列表。

> 为了确保本次 Docker 部署实验环境的纯净，不受上一节操作的影响，笔者已将云服务器重装为全新的 Ubuntu 22.04 系统。

完成后，我们就可以执行 Docker 官方提供的一键安装脚本了。这个脚本会自动检测你的 Linux 发行版，并安装最新稳定版的 Docker Engine 和 Docker Compose。

（1）**执行 Docker 官方安装脚本**

这个脚本会自动检测你的 Linux 发行版，并安装最新稳定版的 Docker Engine 和 Docker Compose。

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

> **网络问题提示**
> 在国内的服务器上执行 `curl` 命令时，若遇到 `Connection reset by peer` 等网络错误，**应首先检查云服务器的安全组是否已放行 443 (HTTPS) 端口的出站规则**。如果确认端口已放行但问题依旧，通常是网络链路不稳定所致，此时可以改用国内镜像源（如 DaoCloud）提供的一键安装命令：`curl -sSL https://get.daocloud.io/docker | sh`。

（2）**验证安装**
安装完成后，执行以下命令来检查 Docker 和 Docker Compose 是否成功安装。

```bash
docker --version
docker compose version
```

如果都能如图 14-16 正确输出版本号，说明 Docker 环境已经准备就绪。

<p align="center">
  <img src="./images/14_3_1.png" width="80%" alt="验证 Docker 安装成功" />
  <br />
  <em>图 14-16 验证 Docker 安装成功</em>
</p>

（3）**(可选) 配置国内镜像加速**

默认情况下，Docker 会从国外的 Docker Hub 拉取镜像，速度可能较慢。我们可以配置国内的镜像加速器来提升下载速度。

创建或修改 Docker 的配置文件：

```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

在打开的编辑器中，粘贴以下内容（以阿里云加速器为例），然后保存并退出 (`Ctrl+X`, `Y`, `Enter`)：

```json
{
  "registry-mirrors": ["https://<阿里云镜像 ID>.mirror.aliyuncs.com"]
}
```

> 每个阿里云用户都可以免费获取一个专属的镜像加速器地址。获取步骤如下：登录阿里云控制台后，在顶部搜索框搜索“容器镜像服务”，进入后在如图 14-17 左侧菜单的“镜像工具”下找到“镜像加速器”，即可看到专属地址。如果不想注册阿里云，也可以使用其他公开的加速器，如网易的 `https://hub-mirror.c.163.com`。

<p align="center">
  <img src="./images/14_3_2.png" width="80%" alt="获取阿里云镜像加速器" />
  <br />
  <em>图 14-17 获取阿里云镜像加速器</em>
</p>

最后，重启 Docker 服务使配置生效：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 三、容器化 NER 应用

现在，我们需要为 `ner_deployment` 项目创建两个核心文件，`Dockerfile` 和 `docker-compose.yml`。在项目根目录（即 `code/C14/ner_deployment/`）中创建它们。

### 3.1 编写 Dockerfile

`Dockerfile` 是一个纯文本文件，用于告诉 Docker **如何一步步构建镜像**。它由多条指令组成，每一条指令都会在镜像中生成一层。常见的指令包括：

- **`FROM`**: 指定基础镜像，是整个镜像构建的起点。
- **`WORKDIR`**: 设置容器内的工作目录，后续的文件复制和命令执行都会在这个目录下进行。
- **`COPY` / `ADD`**: 将宿主机上的文件或目录复制到镜像中。
- **`RUN`**: 在构建阶段执行一条命令，通常用于安装依赖或做环境配置。
- **`EXPOSE`**: 声明容器对外暴露的端口（更偏文档作用）。
- **`CMD` / `ENTRYPOINT`**: 定义容器启动时默认要执行的命令。

以本次部署的 NER 项目为例，为它编写一个完整的 `Dockerfile`：

```dockerfile
# ner_deployment/Dockerfile

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 设置 PyPI 镜像源，加速依赖安装
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./

RUN uv pip install --system --no-cache .

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["gunicorn", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
```

这里我们选择了轻量的官方镜像 `python:3.10-slim` 作为基础镜像，并通过 `ENV` 指令关闭 Python 输出缓冲、禁止生成 `.pyc` 文件，让容器中的日志更适合调试、镜像内容更整洁。接着，通过 `WORKDIR /app` 将工作目录固定在 `/app`。考虑到国内网络环境，我们通过 `RUN pip config set` 命令在镜像内持久配置 PyPI 镜像源为阿里云镜像，以解决 `pip` 安装依赖时的网络超时问题。然后先安装 `uv`，再仅复制 `pyproject.toml` 并执行 `uv pip install --system --no-cache .`，根据 `pyproject.toml` 安装当前项目及其依赖——这种“先复制依赖描述文件，再安装依赖”的方式，可以充分利用 Docker 的层缓存，在依赖不变的情况下避免重复下载和安装。

依赖安装完成后，我们再通过 `COPY . .` 将项目全部代码复制进容器，并创建一个名为 `appuser` 的非 root 用户来运行应用，从而降低安全风险。最后，通过 `EXPOSE 8000` 声明服务端口，并在 `CMD` 中使用 `gunicorn -k uvicorn.workers.UvicornWorker` 启动 FastAPI 应用，实现与前文介绍的 `systemd` 方案相同的生产级启动方式，但全部封装在容器内部。

### 3.2 编写 docker-compose.yml

在使用 Docker Compose 时，我们通常通过一个 `docker-compose.yml` 文件来描述整个应用的服务拓扑。该文件至少会包含一个顶层的 `services` 字段，在其下按名称划分出多个服务，每个服务可以配置镜像来源（`image` / `build`）、端口映射（`ports`）、环境变量（`environment`）、重启策略（`restart`）等常用选项。

下面是一个用于当前 NER 服务的 `docker-compose.yml` 示例：

```yaml
# ner_deployment/docker-compose.yml

services:
  ner_api:
    build: .
    container_name: ner_api_service
    ports:
      - "8000:8000"
    restart: always
```

在这个 Compose 文件中，我们只定义了一个名为 `ner_api` 的服务。通过 `build: .` 指定它使用当前目录下的 `Dockerfile` 来构建镜像，而不是直接从远程仓库拉取现成镜像；`container_name: ner_api_service` 则为该容器指定了一个固定且易于识别的名字，便于在后续调试和运维中使用。`ports` 字段中的 `"8000:8000"` 将云服务器的 8000 端口映射到容器内部的 8000 端口，使外部请求可以通过服务器 IP 访问到容器中的 FastAPI 服务；`restart: always` 则为服务设置了自动重启策略，确保在容器异常退出或服务器重启后，NER 接口能自动恢复运行。在大多数单服务场景下，这一行配置基本上可以替代上节中需要手动编写 `systemd` 服务文件来实现的持久化效果，由 Docker 自身接管重启策略，整体更加简洁高效。

## 四、部署与测试

### 4.1 上传项目文件

现在，我们的本地 `ner_deployment` 文件夹中包含了项目源代码，以及刚刚创建的 `Dockerfile` 和 `docker-compose.yml`。像上一节一样，使用 FinalShell 或其他 SFTP 工具，将整个 `ner_deployment` 文件夹上传到云服务器的用户主目录（例如 `/root`）。

### 4.2 检查 Docker 服务状态

在执行 `docker compose` 命令前，我们先确认一下 Docker 服务是否正在运行。

```bash
sudo systemctl status docker
```

如果看到图 14-18 所示绿色的 `active (running)` 字样，说明 Docker 服务正常。如果服务未启动，可以执行 `sudo systemctl start docker` 来启动它。

<p align="center">
  <img src="./images/14_3_3.png" width="80%" alt="检查 Docker 服务状态" />
  <br />
  <em>图 14-18 检查 Docker 服务状态</em>
</p>

### 4.3 构建并启动服务

上传完成后，在服务器的终端中，进入项目目录：

```bash
cd ner_deployment
```

接下来，执行以下命令，Docker Compose 将会读取 `docker-compose.yml` 和 `Dockerfile`，自动完成镜像构建和容器启动：

```bash
sudo docker compose up --build -d
```
- `up`: 启动服务。
- `--build`: 强制重新构建镜像。首次运行时需要，后续如果修改了 `Dockerfile` 或项目代码也需要加上此参数。
- `-d`: detached 模式，让服务在后台运行。

首次执行时，我们会看到 Docker 正在一步步执行 `Dockerfile` 中的指令来构建镜像，然后启动容器。整个过程可能需要几分钟，具体时间取决于服务器网络状况和性能。

<p align="center">
  <img src="./images/14_3_4.png" width="80%" alt="Docker Compose 启动成功" />
  <br />
  <em>图 14-19 Docker Compose 启动成功</em>
</p>

### 4.4 管理服务

服务启动后，可以使用以下命令来管理它：

- **查看服务状态和日志**:

  ```bash
  # 查看正在运行的容器
  sudo docker compose ps
  
  # 实时查看服务日志
  sudo docker compose logs -f
  ```

  <p align="center">
    <img src="./images/14_3_5.png" width="80%" alt="查看服务状态与日志" />
    <br />
    <em>图 14-20 查看服务状态与日志</em>
  </p>

- **停止服务**:

  ```bash
  sudo docker compose down
  ```
  此命令会停止并移除由该 `docker-compose.yml` 文件创建的容器和网络。

### 4.5 测试云端服务

不出意外的话，我们的 NER API 服务已经通过 Docker 在云端成功运行。除了使用 `curl`，我们还可以利用 FastAPI 自动生成的交互式 API 文档来进行测试，这种方式更加直观。

在浏览器中打开 `http://<服务器公网IP>:8000/docs`，会看到如图 14-21 所示的页面。

<p align="center">
  <img src="./images/14_3_6.png" width="80%" alt="FastAPI 交互式 API 文档" />
  <br />
  <em>图 14-21 FastAPI 交互式 API 文档</em>
</p>

展开 `/predict/ner` 接口，点击右上角的 "Try it out" 按钮。

<p align="center">
  <img src="./images/14_3_7.png" width="80%" alt="进入接口测试视图" />
  <br />
  <em>图 14-22 进入接口测试视图</em>
</p>

然后如图 14-23 在 "Request body" 中输入待识别的文本“患者自述发热、咳嗽，伴有轻微头痛。”，最后点击蓝色的 "Execute" 按钮执行请求。

<p align="center">
  <img src="./images/14_3_8.png" width="80%" alt="在 API 文档中发起请求" />
  <br />
  <em>图 14-23 在 API 文档中发起请求</em>
</p>

页面下方会立刻显示出服务器的响应结果。如图 14-24 所示，我们成功收到了包含实体识别结果的 JSON 数据，说明我们的 NER 模型服务已通过 Docker 成功部署并能正常工作。

<p align="center">
  <img src="./images/14_3_9.png" width="80%" alt="成功获取模型预测结果" />
  <br />
  <em>图 14-24 成功获取模型预测结果</em>
</p>

## 本章小结

至此，我们系统地走过了将一个训练好的 NER 模型部署为生产级服务的完整流程。首先利用现代化的高性能 Web 框架 **FastAPI**，为模型构建了一个功能完备、自带交互式文档的 API 接口。随后，我们实践了第一种云端部署方案。在服务器上手动配置环境，使用新兴的 Python 打包工具 **`uv`** 进行依赖管理，并借助 Linux 系统的 **`systemd`** 服务来确保应用持久化运行。这套流程代表了一种完整且实用的传统服务器部署方法。

在此基础上，我们探索了更先进的**容器化**部署方案。通过编写 `Dockerfile` 和 `docker-compose.yml`，将应用及其所有依赖打包成一个标准化的、与环境无关的 Docker 镜像，并利用 **Docker Compose** 来优雅地定义和管理服务。这种“一次构建，随处运行”的模式，不仅以更简洁的 `restart: always` 策略替代了 `systemd`，还从根本上解决了环境不一致的难题，是当前业界推崇的主流部署方案。通过本章的学习，我们不仅掌握了两种模型服务部署方法，更体会了从传统部署到现代化容器化部署的演进。