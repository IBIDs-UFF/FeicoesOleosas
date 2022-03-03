# Feições Oleosas

A solução *Feições Oleosas* é composto por três componentes:

1. Scripts [Python](https://www.python.org/) para processamento de vídeo (pasta `py`);
2. Servidor web para ativar o processamento de vídeo a partir de um browser ou do aplicativo (pasta `http`); e
3. Aplicativo [Android](https://www.android.com/) para realizar a captura de vídeos e enviá-los para processamento no servidor web (pasta `app`).

A forma mais simples de executar a solução é utilizando [Docker](https://www.docker.com/) para levantar o servidor web que dispara o processamento e um smartphone para executar o aplicativo. Com Docker, todas as ferramentas e dependências necessárias são instaladas automaticamente e ficam autocontidas em um container, não interferindo no funcionamento do restante do seu sistema. Logo, é preciso que o Docker esteja disponível no computador que será ao mesmo tempo o servidor web (componente 2) e de processamento (componente 1). Para melhor desempenho, este servidor deve estar equipado com uma Unidade de Processamento Gráfico (GPU, do inglês *Graphics Processing Unit*) com suporte a [CUDA](https://developer.nvidia.com/cuda-gpus).

Você também precisará do [Git](https://git-scm.com/) para baixar o código fonte da solução.

Siga os passos descritos a seguir para realizar as instalações necessárias.

## Instalação

Primeiro, instale o Git em seu sistema seguindo as instruções disponíveis em [https://github.com/git-guides/install-git/](https://github.com/git-guides/install-git/). 

Agora, instale o Docker Engine em seu sistema seguindo as instruções disponíveis em [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/).

Utilizando o PowerShell do Microsoft Windows ou um terminal Linux, execute:

```bash
git clone https://github.com/IBIDs-UFF/FeicoesOleosas
cd FeicoesOleosas
docker build -t feicoesoleosas .
```

O primeiro comando criará a pasta `FeicoesOleosas` na pasta corrente e colocará dentro dela a versão mais recente do código fonte da solução *Feições Oleosas*. O segundo comando fará com que a pasta `FeicoesOleosas` seja a pasta corrente. Por sua vez, o terceiro comando criará o container do Docker contendo tudo o que é necessário para inicializar o servidor web e de processamento. O tempo de criação do container é longo. Felizmente, a criação só é preciso ser executada uma vez.

## Inicialização do Servidor

Uma vez que o container estiver criado, inicialize o servidor executando o comando abaixo, substituindo `<port>` pela porta liberada pelo administrador da rede:

```bash
docker run feicoesoleosas python wd/http/server.py <port>
```

A partir de agora o servidor web e de processamento estará disponível no endereço `<ip-address>:<port>`, onde `<ip-address>` é o [endereço IP](https://en.wikipedia.org/wiki/IP_address) da máquina que hospeda o servidor. Ou seja, após inicializado o servidor poderá ser acessado por um browser ou pelo aplicativo. Recomenda-se o uso do browser [Chrome](https://www.google.com/chrome/).

Por exemplo, da própria máquina que atua como servidor, se `<port>` é igual a `8000` então o endereço do servidor web é [http://127.0.0.1:8000](http://127.0.0.1:8000) ou [http://localhost:8000](http://localhost:8000).

O endereço de acesso a partir de outra máquina e do aplicativo dependerá do endereço IP pelo qual a máquina é visível e/ou do nome de domínio atribuído pelo [Sistema de Nome de Domínio (DNS)](https://en.wikipedia.org/wiki/Domain_Name_System) para torná-la visível. Por exemplo, se o endereço IP da máquina é `192.0.2.44`, o nome de domínio é `www.feicesoleosas.com` e a porta é `8000` então o servidor web é acessado por [http://192.0.2.44:8000](http://192.0.2.44:8000) e/ou [http://www.feicesoleosas.com:8000](http://www.feicesoleosas.com:8000).

## Instalação do Aplicativo

A última versão do aplicativo compilada pelo desenvolvedor está disponível na [Google Play Store](https://play.google.com/store), no endereço [TODO](TODO). A forma mais simples de obtê-lo é baixando e instalar o aplicativo diretamente de um smartphone Android.

Utilize o [Android Studio](https://developer.android.com/studio/) para abrir e compilar o projeto presente na pasta `app`. Esta ação só é necessária caso alguma alteração tenha sido feita no código fonte presente nesta pasta.

O uso do aplicativo é bastante intuitivo. A interface o guiará para executar as ações de captura e processamento de vídeos. Mas atenção, para que o processamento seja feito é preciso que o servidor web esteja ativo.