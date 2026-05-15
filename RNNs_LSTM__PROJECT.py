{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP28IUeZtLdc4r1hMYF0red",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Prajwalpraju17/DeepLearning-HandsOn/blob/main/RNNs_LSTM__PROJECT.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hcCtHpUcXdty"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from keras.models import Sequential\n",
        "import datetime as dt\n",
        "import tensorflow\n",
        "from keras.layers import Dense,Dropout,LSTM\n",
        "from sklearn.preprocessing import MinMaxScaler"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "s1=[0,1,1,2,3,5,8,13,21,34,55,89]\n",
        "s1=[1/(1+np.exp(-x)) for x in np.arange(-6,6,0.1)]"
      ],
      "metadata": {
        "id": "9Ih-aJWXYqPC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "n_steps=2\n",
        "n_f1=1"
      ],
      "metadata": {
        "id": "Yc4_yNZaZZaB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "n_steps"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cY2b-nRGZdXY",
        "outputId": "22c43063-7692-45e7-9712-7ddf3984a510"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(s1)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "1NaPRoNKaMO0",
        "outputId": "0785bd5f-5d6d-4cc0-b607-16d8f9f7bc4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "len(s1)-n_steps"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5zk3_wa9aUvd",
        "outputId": "dde6a760-1545-484b-8218-a1501556eeb0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "118"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=[]\n",
        "y=[]\n",
        "for i in range(len(s1)-n_steps):\n",
        "  x.append(s1[i:i+n_steps])\n",
        "  y.append(s1[i+n_steps])\n",
        "x=np.array(x)\n",
        "y=np.array(y)"
      ],
      "metadata": {
        "id": "kD2_meCUaf2c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wBU_zoS6ayip",
        "outputId": "557d0ce4-f925-4f15-b0a6-5e0223aac90e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.00247262, 0.00273196],\n",
              "       [0.00273196, 0.00301842],\n",
              "       [0.00301842, 0.00333481],\n",
              "       [0.00333481, 0.00368424],\n",
              "       [0.00368424, 0.00407014],\n",
              "       [0.00407014, 0.00449627],\n",
              "       [0.00449627, 0.0049668 ],\n",
              "       [0.0049668 , 0.0054863 ],\n",
              "       [0.0054863 , 0.0060598 ],\n",
              "       [0.0060598 , 0.00669285],\n",
              "       [0.00669285, 0.00739154],\n",
              "       [0.00739154, 0.00816257],\n",
              "       [0.00816257, 0.0090133 ],\n",
              "       [0.0090133 , 0.0099518 ],\n",
              "       [0.0099518 , 0.01098694],\n",
              "       [0.01098694, 0.01212843],\n",
              "       [0.01212843, 0.01338692],\n",
              "       [0.01338692, 0.01477403],\n",
              "       [0.01477403, 0.0163025 ],\n",
              "       [0.0163025 , 0.01798621],\n",
              "       [0.01798621, 0.01984031],\n",
              "       [0.01984031, 0.02188127],\n",
              "       [0.02188127, 0.02412702],\n",
              "       [0.02412702, 0.02659699],\n",
              "       [0.02659699, 0.02931223],\n",
              "       [0.02931223, 0.03229546],\n",
              "       [0.03229546, 0.03557119],\n",
              "       [0.03557119, 0.03916572],\n",
              "       [0.03916572, 0.04310725],\n",
              "       [0.04310725, 0.04742587],\n",
              "       [0.04742587, 0.05215356],\n",
              "       [0.05215356, 0.05732418],\n",
              "       [0.05732418, 0.06297336],\n",
              "       [0.06297336, 0.06913842],\n",
              "       [0.06913842, 0.07585818],\n",
              "       [0.07585818, 0.0831727 ],\n",
              "       [0.0831727 , 0.09112296],\n",
              "       [0.09112296, 0.09975049],\n",
              "       [0.09975049, 0.10909682],\n",
              "       [0.10909682, 0.11920292],\n",
              "       [0.11920292, 0.13010847],\n",
              "       [0.13010847, 0.14185106],\n",
              "       [0.14185106, 0.15446527],\n",
              "       [0.15446527, 0.16798161],\n",
              "       [0.16798161, 0.18242552],\n",
              "       [0.18242552, 0.19781611],\n",
              "       [0.19781611, 0.21416502],\n",
              "       [0.21416502, 0.23147522],\n",
              "       [0.23147522, 0.24973989],\n",
              "       [0.24973989, 0.26894142],\n",
              "       [0.26894142, 0.2890505 ],\n",
              "       [0.2890505 , 0.31002552],\n",
              "       [0.31002552, 0.33181223],\n",
              "       [0.33181223, 0.35434369],\n",
              "       [0.35434369, 0.37754067],\n",
              "       [0.37754067, 0.40131234],\n",
              "       [0.40131234, 0.42555748],\n",
              "       [0.42555748, 0.450166  ],\n",
              "       [0.450166  , 0.47502081],\n",
              "       [0.47502081, 0.5       ],\n",
              "       [0.5       , 0.52497919],\n",
              "       [0.52497919, 0.549834  ],\n",
              "       [0.549834  , 0.57444252],\n",
              "       [0.57444252, 0.59868766],\n",
              "       [0.59868766, 0.62245933],\n",
              "       [0.62245933, 0.64565631],\n",
              "       [0.64565631, 0.66818777],\n",
              "       [0.66818777, 0.68997448],\n",
              "       [0.68997448, 0.7109495 ],\n",
              "       [0.7109495 , 0.73105858],\n",
              "       [0.73105858, 0.75026011],\n",
              "       [0.75026011, 0.76852478],\n",
              "       [0.76852478, 0.78583498],\n",
              "       [0.78583498, 0.80218389],\n",
              "       [0.80218389, 0.81757448],\n",
              "       [0.81757448, 0.83201839],\n",
              "       [0.83201839, 0.84553473],\n",
              "       [0.84553473, 0.85814894],\n",
              "       [0.85814894, 0.86989153],\n",
              "       [0.86989153, 0.88079708],\n",
              "       [0.88079708, 0.89090318],\n",
              "       [0.89090318, 0.90024951],\n",
              "       [0.90024951, 0.90887704],\n",
              "       [0.90887704, 0.9168273 ],\n",
              "       [0.9168273 , 0.92414182],\n",
              "       [0.92414182, 0.93086158],\n",
              "       [0.93086158, 0.93702664],\n",
              "       [0.93702664, 0.94267582],\n",
              "       [0.94267582, 0.94784644],\n",
              "       [0.94784644, 0.95257413],\n",
              "       [0.95257413, 0.95689275],\n",
              "       [0.95689275, 0.96083428],\n",
              "       [0.96083428, 0.96442881],\n",
              "       [0.96442881, 0.96770454],\n",
              "       [0.96770454, 0.97068777],\n",
              "       [0.97068777, 0.97340301],\n",
              "       [0.97340301, 0.97587298],\n",
              "       [0.97587298, 0.97811873],\n",
              "       [0.97811873, 0.98015969],\n",
              "       [0.98015969, 0.98201379],\n",
              "       [0.98201379, 0.9836975 ],\n",
              "       [0.9836975 , 0.98522597],\n",
              "       [0.98522597, 0.98661308],\n",
              "       [0.98661308, 0.98787157],\n",
              "       [0.98787157, 0.98901306],\n",
              "       [0.98901306, 0.9900482 ],\n",
              "       [0.9900482 , 0.9909867 ],\n",
              "       [0.9909867 , 0.99183743],\n",
              "       [0.99183743, 0.99260846],\n",
              "       [0.99260846, 0.99330715],\n",
              "       [0.99330715, 0.9939402 ],\n",
              "       [0.9939402 , 0.9945137 ],\n",
              "       [0.9945137 , 0.9950332 ],\n",
              "       [0.9950332 , 0.99550373],\n",
              "       [0.99550373, 0.99592986],\n",
              "       [0.99592986, 0.99631576],\n",
              "       [0.99631576, 0.99666519],\n",
              "       [0.99666519, 0.99698158]])"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(s1)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "MphcWG1dbEAH",
        "outputId": "e5b3b844-07d4-46b2-eec8-9fd3593e43c7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 448
        },
        "id": "vW1LuQt1dteS",
        "outputId": "321edba1-b214-497c-884f-1ee1e0be170d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7c81e5392810>]"
            ]
          },
          "metadata": {},
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qTuBlaC6gI2h",
        "outputId": "682af154-6a4d-4bdb-f756-eff18839c25f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(118, 2)"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=x.reshape((X.shape[0],X.shape[1],n_f1))\n",
        "x.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6qTq_QJ1gYVO",
        "outputId": "859dbc52-70f5-4c15-ad89-639f3eb32315"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(118, 2, 1)"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rObXHFZThQxt",
        "outputId": "12367235-0235-44c3-d8df-3b0e666a6253"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[[0.00247262],\n",
              "        [0.00273196]],\n",
              "\n",
              "       [[0.00273196],\n",
              "        [0.00301842]],\n",
              "\n",
              "       [[0.00301842],\n",
              "        [0.00333481]],\n",
              "\n",
              "       [[0.00333481],\n",
              "        [0.00368424]],\n",
              "\n",
              "       [[0.00368424],\n",
              "        [0.00407014]],\n",
              "\n",
              "       [[0.00407014],\n",
              "        [0.00449627]],\n",
              "\n",
              "       [[0.00449627],\n",
              "        [0.0049668 ]],\n",
              "\n",
              "       [[0.0049668 ],\n",
              "        [0.0054863 ]],\n",
              "\n",
              "       [[0.0054863 ],\n",
              "        [0.0060598 ]],\n",
              "\n",
              "       [[0.0060598 ],\n",
              "        [0.00669285]],\n",
              "\n",
              "       [[0.00669285],\n",
              "        [0.00739154]],\n",
              "\n",
              "       [[0.00739154],\n",
              "        [0.00816257]],\n",
              "\n",
              "       [[0.00816257],\n",
              "        [0.0090133 ]],\n",
              "\n",
              "       [[0.0090133 ],\n",
              "        [0.0099518 ]],\n",
              "\n",
              "       [[0.0099518 ],\n",
              "        [0.01098694]],\n",
              "\n",
              "       [[0.01098694],\n",
              "        [0.01212843]],\n",
              "\n",
              "       [[0.01212843],\n",
              "        [0.01338692]],\n",
              "\n",
              "       [[0.01338692],\n",
              "        [0.01477403]],\n",
              "\n",
              "       [[0.01477403],\n",
              "        [0.0163025 ]],\n",
              "\n",
              "       [[0.0163025 ],\n",
              "        [0.01798621]],\n",
              "\n",
              "       [[0.01798621],\n",
              "        [0.01984031]],\n",
              "\n",
              "       [[0.01984031],\n",
              "        [0.02188127]],\n",
              "\n",
              "       [[0.02188127],\n",
              "        [0.02412702]],\n",
              "\n",
              "       [[0.02412702],\n",
              "        [0.02659699]],\n",
              "\n",
              "       [[0.02659699],\n",
              "        [0.02931223]],\n",
              "\n",
              "       [[0.02931223],\n",
              "        [0.03229546]],\n",
              "\n",
              "       [[0.03229546],\n",
              "        [0.03557119]],\n",
              "\n",
              "       [[0.03557119],\n",
              "        [0.03916572]],\n",
              "\n",
              "       [[0.03916572],\n",
              "        [0.04310725]],\n",
              "\n",
              "       [[0.04310725],\n",
              "        [0.04742587]],\n",
              "\n",
              "       [[0.04742587],\n",
              "        [0.05215356]],\n",
              "\n",
              "       [[0.05215356],\n",
              "        [0.05732418]],\n",
              "\n",
              "       [[0.05732418],\n",
              "        [0.06297336]],\n",
              "\n",
              "       [[0.06297336],\n",
              "        [0.06913842]],\n",
              "\n",
              "       [[0.06913842],\n",
              "        [0.07585818]],\n",
              "\n",
              "       [[0.07585818],\n",
              "        [0.0831727 ]],\n",
              "\n",
              "       [[0.0831727 ],\n",
              "        [0.09112296]],\n",
              "\n",
              "       [[0.09112296],\n",
              "        [0.09975049]],\n",
              "\n",
              "       [[0.09975049],\n",
              "        [0.10909682]],\n",
              "\n",
              "       [[0.10909682],\n",
              "        [0.11920292]],\n",
              "\n",
              "       [[0.11920292],\n",
              "        [0.13010847]],\n",
              "\n",
              "       [[0.13010847],\n",
              "        [0.14185106]],\n",
              "\n",
              "       [[0.14185106],\n",
              "        [0.15446527]],\n",
              "\n",
              "       [[0.15446527],\n",
              "        [0.16798161]],\n",
              "\n",
              "       [[0.16798161],\n",
              "        [0.18242552]],\n",
              "\n",
              "       [[0.18242552],\n",
              "        [0.19781611]],\n",
              "\n",
              "       [[0.19781611],\n",
              "        [0.21416502]],\n",
              "\n",
              "       [[0.21416502],\n",
              "        [0.23147522]],\n",
              "\n",
              "       [[0.23147522],\n",
              "        [0.24973989]],\n",
              "\n",
              "       [[0.24973989],\n",
              "        [0.26894142]],\n",
              "\n",
              "       [[0.26894142],\n",
              "        [0.2890505 ]],\n",
              "\n",
              "       [[0.2890505 ],\n",
              "        [0.31002552]],\n",
              "\n",
              "       [[0.31002552],\n",
              "        [0.33181223]],\n",
              "\n",
              "       [[0.33181223],\n",
              "        [0.35434369]],\n",
              "\n",
              "       [[0.35434369],\n",
              "        [0.37754067]],\n",
              "\n",
              "       [[0.37754067],\n",
              "        [0.40131234]],\n",
              "\n",
              "       [[0.40131234],\n",
              "        [0.42555748]],\n",
              "\n",
              "       [[0.42555748],\n",
              "        [0.450166  ]],\n",
              "\n",
              "       [[0.450166  ],\n",
              "        [0.47502081]],\n",
              "\n",
              "       [[0.47502081],\n",
              "        [0.5       ]],\n",
              "\n",
              "       [[0.5       ],\n",
              "        [0.52497919]],\n",
              "\n",
              "       [[0.52497919],\n",
              "        [0.549834  ]],\n",
              "\n",
              "       [[0.549834  ],\n",
              "        [0.57444252]],\n",
              "\n",
              "       [[0.57444252],\n",
              "        [0.59868766]],\n",
              "\n",
              "       [[0.59868766],\n",
              "        [0.62245933]],\n",
              "\n",
              "       [[0.62245933],\n",
              "        [0.64565631]],\n",
              "\n",
              "       [[0.64565631],\n",
              "        [0.66818777]],\n",
              "\n",
              "       [[0.66818777],\n",
              "        [0.68997448]],\n",
              "\n",
              "       [[0.68997448],\n",
              "        [0.7109495 ]],\n",
              "\n",
              "       [[0.7109495 ],\n",
              "        [0.73105858]],\n",
              "\n",
              "       [[0.73105858],\n",
              "        [0.75026011]],\n",
              "\n",
              "       [[0.75026011],\n",
              "        [0.76852478]],\n",
              "\n",
              "       [[0.76852478],\n",
              "        [0.78583498]],\n",
              "\n",
              "       [[0.78583498],\n",
              "        [0.80218389]],\n",
              "\n",
              "       [[0.80218389],\n",
              "        [0.81757448]],\n",
              "\n",
              "       [[0.81757448],\n",
              "        [0.83201839]],\n",
              "\n",
              "       [[0.83201839],\n",
              "        [0.84553473]],\n",
              "\n",
              "       [[0.84553473],\n",
              "        [0.85814894]],\n",
              "\n",
              "       [[0.85814894],\n",
              "        [0.86989153]],\n",
              "\n",
              "       [[0.86989153],\n",
              "        [0.88079708]],\n",
              "\n",
              "       [[0.88079708],\n",
              "        [0.89090318]],\n",
              "\n",
              "       [[0.89090318],\n",
              "        [0.90024951]],\n",
              "\n",
              "       [[0.90024951],\n",
              "        [0.90887704]],\n",
              "\n",
              "       [[0.90887704],\n",
              "        [0.9168273 ]],\n",
              "\n",
              "       [[0.9168273 ],\n",
              "        [0.92414182]],\n",
              "\n",
              "       [[0.92414182],\n",
              "        [0.93086158]],\n",
              "\n",
              "       [[0.93086158],\n",
              "        [0.93702664]],\n",
              "\n",
              "       [[0.93702664],\n",
              "        [0.94267582]],\n",
              "\n",
              "       [[0.94267582],\n",
              "        [0.94784644]],\n",
              "\n",
              "       [[0.94784644],\n",
              "        [0.95257413]],\n",
              "\n",
              "       [[0.95257413],\n",
              "        [0.95689275]],\n",
              "\n",
              "       [[0.95689275],\n",
              "        [0.96083428]],\n",
              "\n",
              "       [[0.96083428],\n",
              "        [0.96442881]],\n",
              "\n",
              "       [[0.96442881],\n",
              "        [0.96770454]],\n",
              "\n",
              "       [[0.96770454],\n",
              "        [0.97068777]],\n",
              "\n",
              "       [[0.97068777],\n",
              "        [0.97340301]],\n",
              "\n",
              "       [[0.97340301],\n",
              "        [0.97587298]],\n",
              "\n",
              "       [[0.97587298],\n",
              "        [0.97811873]],\n",
              "\n",
              "       [[0.97811873],\n",
              "        [0.98015969]],\n",
              "\n",
              "       [[0.98015969],\n",
              "        [0.98201379]],\n",
              "\n",
              "       [[0.98201379],\n",
              "        [0.9836975 ]],\n",
              "\n",
              "       [[0.9836975 ],\n",
              "        [0.98522597]],\n",
              "\n",
              "       [[0.98522597],\n",
              "        [0.98661308]],\n",
              "\n",
              "       [[0.98661308],\n",
              "        [0.98787157]],\n",
              "\n",
              "       [[0.98787157],\n",
              "        [0.98901306]],\n",
              "\n",
              "       [[0.98901306],\n",
              "        [0.9900482 ]],\n",
              "\n",
              "       [[0.9900482 ],\n",
              "        [0.9909867 ]],\n",
              "\n",
              "       [[0.9909867 ],\n",
              "        [0.99183743]],\n",
              "\n",
              "       [[0.99183743],\n",
              "        [0.99260846]],\n",
              "\n",
              "       [[0.99260846],\n",
              "        [0.99330715]],\n",
              "\n",
              "       [[0.99330715],\n",
              "        [0.9939402 ]],\n",
              "\n",
              "       [[0.9939402 ],\n",
              "        [0.9945137 ]],\n",
              "\n",
              "       [[0.9945137 ],\n",
              "        [0.9950332 ]],\n",
              "\n",
              "       [[0.9950332 ],\n",
              "        [0.99550373]],\n",
              "\n",
              "       [[0.99550373],\n",
              "        [0.99592986]],\n",
              "\n",
              "       [[0.99592986],\n",
              "        [0.99631576]],\n",
              "\n",
              "       [[0.99631576],\n",
              "        [0.99666519]],\n",
              "\n",
              "       [[0.99666519],\n",
              "        [0.99698158]]])"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model=Sequential()\n",
        "model.add(LSTM(units=2,activation=\"relu\",return_sequences=True,input_shape=(X.shape[1],1)))\n",
        "model.add(LSTM(units=4,activation=\"relu\",return_sequences=False))\n",
        "model.add(Dropout(0.2))\n",
        "model.add(Dense(units=1))\n",
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "id": "RbQ3CjwMhTud",
        "outputId": "106d1e47-66cc-4f68-a446-d0715c929ea3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/rnn/rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1mModel: \"sequential_1\"\u001b[0m\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential_1\"</span>\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ lstm_2 (\u001b[38;5;33mLSTM\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m)           │            \u001b[38;5;34m32\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ lstm_3 (\u001b[38;5;33mLSTM\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m)              │           \u001b[38;5;34m112\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)             │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m)              │             \u001b[38;5;34m0\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m)              │             \u001b[38;5;34m5\u001b[0m │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ lstm_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">LSTM</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>)           │            <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ lstm_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">LSTM</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>)              │           <span style=\"color: #00af00; text-decoration-color: #00af00\">112</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)             │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>)              │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>)              │             <span style=\"color: #00af00; text-decoration-color: #00af00\">5</span> │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m149\u001b[0m (596.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">149</span> (596.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m149\u001b[0m (596.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">149</span> (596.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer=\"adam\",loss=\"mean_squared_error\",metrics=[\"accuracy\"])"
      ],
      "metadata": {
        "id": "FuoTMf3WhcJ6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history=model.fit(x,y,epochs=20,batch_size=2,validation_split=0.1)"
      ],
      "metadata": {
        "id": "uDbAOeGXjt_b",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5d3594cd-805f-4701-af83-ccf8ff1ca83a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 18ms/step - accuracy: 0.0000e+00 - loss: 0.2191 - val_accuracy: 0.0000e+00 - val_loss: 0.5871\n",
            "Epoch 2/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - accuracy: 0.0000e+00 - loss: 0.1950 - val_accuracy: 0.0000e+00 - val_loss: 0.4386\n",
            "Epoch 3/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - accuracy: 0.0000e+00 - loss: 0.1308 - val_accuracy: 0.0000e+00 - val_loss: 0.3227\n",
            "Epoch 4/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0941 - val_accuracy: 0.0000e+00 - val_loss: 0.1914\n",
            "Epoch 5/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.1021 - val_accuracy: 0.0000e+00 - val_loss: 0.1319\n",
            "Epoch 6/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0943 - val_accuracy: 0.0000e+00 - val_loss: 0.1025\n",
            "Epoch 7/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.1081 - val_accuracy: 0.0000e+00 - val_loss: 0.0814\n",
            "Epoch 8/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.1117 - val_accuracy: 0.0000e+00 - val_loss: 0.0645\n",
            "Epoch 9/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0939 - val_accuracy: 0.0000e+00 - val_loss: 0.0529\n",
            "Epoch 10/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0603 - val_accuracy: 0.0000e+00 - val_loss: 0.0489\n",
            "Epoch 11/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0716 - val_accuracy: 0.0000e+00 - val_loss: 0.0342\n",
            "Epoch 12/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0708 - val_accuracy: 0.0000e+00 - val_loss: 0.0320\n",
            "Epoch 13/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0557 - val_accuracy: 0.0000e+00 - val_loss: 0.0303\n",
            "Epoch 14/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0534 - val_accuracy: 0.0000e+00 - val_loss: 0.0233\n",
            "Epoch 15/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0485 - val_accuracy: 0.0000e+00 - val_loss: 0.0181\n",
            "Epoch 16/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0544 - val_accuracy: 0.0000e+00 - val_loss: 0.0221\n",
            "Epoch 17/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0809 - val_accuracy: 0.0000e+00 - val_loss: 0.0186\n",
            "Epoch 18/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0545 - val_accuracy: 0.0000e+00 - val_loss: 0.0149\n",
            "Epoch 19/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0215 - val_accuracy: 0.0000e+00 - val_loss: 0.0194\n",
            "Epoch 20/20\n",
            "\u001b[1m53/53\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.0000e+00 - loss: 0.0430 - val_accuracy: 0.0000e+00 - val_loss: 0.0119\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(history.history[\"loss\"],label=[\"Training Loss\"])\n",
        "plt.plot(history.history[\"val_loss\"],label=\"Validation Loss\")\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "hv-xYQTvP7IV",
        "outputId": "5f769142-d941-4663-9238-49db407e7e9c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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/SkxMTF899135ObmVhtfUiUsLIz58+cTHx/Pli1buO+++wgLC+N3v/sdEyZMYOvWrSxevJhly5YBEBERcc45CgsLGTlyJMOGDWP9+vVkZWVx77338uCDD1YLXMuXLycuLo7ly5ezd+9eJkyYwIABA7jvvvvOez01XV9VEFm5ciUVFRVMnTqVCRMmsGLFCgAmTpzIwIEDefXVV7Hb7WzevBk/Pz8Apk6dSllZGatWrSIkJITt27cTGhrqdh31pTBytuA2ED8Qjm4yW0cG3G51RSIijae8CP4cb817//4o+IfUa9e7776b559/npUrV3LVVVcBZhfNzTffTEREBBERETzyyCOu/adNm8aSJUt4//336xVGli1bxs6dO1myZAnx8ea/x5///Odzxnk8/vjjrvWkpCQeeeQRFixYwO9+9zuCgoIIDQ3F19eX2Nja/8f13//+NyUlJbzzzjuEhJjXP2fOHMaOHctzzz1HTIz5PLSoqCjmzJmD3W6nZ8+ejBkzhtTU1AaFkdTUVLZs2cKBAwdITEwE4J133qFPnz6sX7+eIUOGkJ6ezv/8z//Qs2dPALp16+Y6Pj09nZtvvpm+ffsC0LlzZ7drcIe6aWri6qpJtbYOEZFWqmfPngwfPpw333wTgL179/L1119zzz33AOBwOHjmmWfo27cvbdq0ITQ0lCVLlpCenl6v8+/YsYPExERXEAEYNmzYOfstXLiQyy67jNjYWEJDQ3n88cfr/R5nvlf//v1dQQTgsssuw+l0smvXLte2Pn36YLfbXd/HxcWRlZXl1nud+Z6JiYmuIALQu3dvIiMj2bFjBwDTp0/n3nvvJSUlhWeffZZ9+/a59n3ooYf44x//yGWXXcasWbMaNGDYHWoZqUmXa+DrF8yWEacTfJTZRKSF8As2Wyisem833HPPPUybNo25c+fy1ltv0aVLF6688koAnn/+ef73f/+Xl156ib59+xISEsLDDz9MWVlZo5W7du1aJk6cyFNPPcXIkSOJiIhgwYIFvPjii432Hmeq6iKpYrPZcDqdTfJeYN4J9POf/5wvvviC//73v8yaNYsFCxZw0003ce+99zJy5Ei++OILvvzyS2bPns2LL77ItGnTmqQW/ZWtSYch4B8KRdmQucXqakREGo/NZnaVWLHUY7zImW677TZ8fHz497//zTvvvMPdd9/tGj+yevVq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K65ocLEtRo/R4BsEpw7A0U1WVyMiImIpt8PIX//6V+677z6mTJlC7969ee211wgODubNN9+s9RiHw8HEiRN56qmn6Ny58wUV3CIEhELPG8x1ddWIiEgr51YYKSsrY+PGjaSkpJw+gY8PKSkprF27ttbjnn76adq3b88999xTr/cpLS0lLy+v2tLiVHXVbP0POB3W1iIiImIht8JIdnY2DoeDmJiYattjYmLIyMio8ZhvvvmGf/zjH8ybN6/e7zN79mwiIiJcS2Jiojtleocu10JgJBRkwsGvra5GRETEMk16N01+fj533nkn8+bNIzo6ut7HzZw5k9zcXNdy6NChJqzSIr7+0Hucua6uGhERacV83dk5Ojoau91OZmZmte2ZmZnExsaes/++ffs4ePAgY8eOdW1zOp3mG/v6smvXLrp06XLOcQEBAQQEBLhTmnfqeytsehu2fwZjXgTfVnDNIiIiZ3GrZcTf359BgwaRmprq2uZ0OklNTWXYsGHn7N+zZ0+2bNnC5s2bXcuNN97I1VdfzebNm1tm94s7Og6HsHgozYU9S62uRkRExBJutYwATJ8+ncmTJzN48GCGDh3KSy+9RGFhIVOmTAFg0qRJJCQkMHv2bAIDA7n44ourHR8ZGQlwzvZWyccOF/8U1s4x5xzp9ROrKxIREWl2boeRCRMmcPz4cZ588kkyMjIYMGAAixcvdg1qTU9Px8dHE7vWW99bzDCyezGU5kNAmNUViYiINCubYXj+FKB5eXlERESQm5tLeHi41eU0LsOAOYPhxF646XXo/zOrKxIREWkU9f37rSYMq9ls1aeHFxERaWUURjzBxbeYX/cth8Jsa2sRERFpZgojniC6K8QNAMMB2z62uhoREZFmpTDiKVxdNZoATUREWheFEU9x8U8BGxz6FnLSra5GRESk2SiMeIrweEi63Fzf+h9raxEREWlGCiOepG/lQFZ11YiISCuiMOJJet0IPn6QuRUyt1tdjYiISLNQGPEkwW2g23Xm+la1joiISOugMOJpzuyq8fzJcUVERC6Ywoin6T4a/EIgJw0Ob7C6GhERkSanMOJp/IOh5xhzXdPDi4hIK6Aw4omqJkDb9hE4KqytRUREpIkpjHiiLldDUBsoPA4HV1ldjYiISJNSGPFEdj/oM95c15wjIiLSwimMeKqqrprtn0F5sbW1iIiINCGF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xQrat29Pjx49eOCBBzhx4kSt+7akzzUzM5MvvviCe+6557z7eutn2xAtNoxkZ2fjcDiIiYmptj0mJoaMjIwaj8nIyHBrf0/ldDp5+OGHueyyy7j44otr3a9Hjx68+eabfPrpp7z77rs4nU6GDx/O4cOHm7Fa9yUnJzN//nwWL17Mq6++yoEDB7jiiivIz8+vcf+W8rl+8skn5OTkcNddd9W6j7d+pjWp+nzc+ewa8nvviUpKSpgxYwa33357nQ9Rc/d3wVOMGjWKd955h9TUVJ577jlWrlzJ6NGjcTgcNe7fUj5XgLfffpuwsDB++tOf1rmft362DeUVT+0V90ydOpWtW7eet39x2LBhDBs2zPX98OHD6dWrF6+//jrPPPNMU5fZYKNHj3at9+vXj+TkZDp27Mj7779fr//b8Fb/+Mc/GD16NPHx8bXu462fqZxWXl7ObbfdhmEYvPrqq3Xu662/Cz/72c9c63379qVfv3506dKFFStWcO2111pYWdN78803mThx4nkHlnvrZ9tQLbZlJDo6GrvdTmZmZrXtmZmZxMbG1nhMbGysW/t7ogcffJDPP/+c5cuX06FDB7eO9fPzY+DAgezdu7eJqmsakZGRdO/evda6W8LnmpaWxrJly7j33nvdOs5bP1PA9fm489k15Pfek1QFkbS0NJYuXer2o+XP97vgqTp37kx0dHStdXv751rl66+/ZteuXW7/HoP3frb11WLDiL+/P4MGDSI1NdW1zel0kpqaWu3/HM80bNiwavsDLF26tNb9PYlhGDz44IN8/PHHfPXVV3Tq1MntczgcDrZs2UJcXFwTVNh0CgoK2LdvX611e/PnWuWtt96iffv2jBkzxq3jvPUzBejUqROxsbHVPru8vDy+++67Wj+7hvzee4qqILJnzx6WLVtG27Zt3T7H+X4XPNXhw4c5ceJErXV78+d6pn/84x8MGjSI/v37u32st3629Wb1CNqmtGDBAiMgIMCYP3++sX37duP+++83IiMjjYyMDMMwDOPOO+80Hn30Udf+q1evNnx9fY0XXnjB2LFjhzFr1izDz8/P2LJli1WXUG8PPPCAERERYaxYscI4duyYaykqKnLtc/b1PvXUU8aSJUuMffv2GRs3bjR+9rOfGYGBgca2bdusuIR6++1vf2usWLHCOHDggLF69WojJSXFiI6ONrKysgzDaFmfq2GYdw1cdNFFxowZM855zds/0/z8fOP77783vv/+ewMw/vrXvxrff/+96w6SZ5991oiMjDQ+/fRT48cffzTGjRtndOrUySguLnad45prrjFeeeUV1/fn+723Sl3XWlZWZtx4441Ghw4djM2bN1f7HS4tLXWd4+xrPd/vglXqutb8/HzjkUceMdauXWscOHDAWLZsmXHJJZcY3bp1M0pKSlzn8JbP1TDO/3NsGIaRm5trBAcHG6+++mqN5/CWz7aptOgwYhiG8corrxgXXXSR4e/vbwwdOtT49ttvXa9deeWVxuTJk6vt//777xvdu3c3/P39jT59+hhffPFFM1fcMECNy1tvveXa5+zrffjhh13/NjExMcYNN9xgbNq0qfmLd9OECROMuLg4w9/f30hISDAmTJhg7N271/V6S/pcDcMwlixZYgDGrl27znnN2z/T5cuX1/hzW3VNTqfTeOKJJ4yYmBgjICDAuPbaa8/5d+jYsaMxa9asatvq+r23Sl3XeuDAgVp/h5cvX+46x9nXer7fBavUda1FRUXG9ddfb7Rr187w8/MzOnbsaNx3333nhApv+VwN4/w/x4ZhGK+//roRFBRk5OTk1HgOb/lsm4rNMAyjSZteREREROrQYseMiIiIiHdQGBERERFLKYyIiIiIpRRGRERExFIKIyIiImIphRERERGxlMKIiIiIWEphRERERCylMCIiIiKWUhgRERERSymMiIiIiKUURkRERMRS/x9IKCgwP0jy8wAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(model.predict(x))\n",
        "plt.plot(y)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 448
        },
        "id": "NBRo-6bQRd4d",
        "outputId": "17519918-f91e-400a-8e7f-9b6c48aff216"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step \n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiUAAAGdCAYAAADNHANuAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAUBVJREFUeJzt3Xd8FHX+x/HX7ibZ9E4aCVWkNykxgB3FBmdH5QSxnR4qgg1U8DxPsYs/G8rZG5bzbCiCKCiC9N57aAkJkGx62Z3fHxOCOVoCyc4meT8fj3nMd2dnZz4ZyebtzHe+YzMMw0BERETEYnarCxAREREBhRIRERHxEQolIiIi4hMUSkRERMQnKJSIiIiIT1AoEREREZ+gUCIiIiI+QaFEREREfIKf1QVUh8fjYffu3YSFhWGz2awuR0RERKrBMAzy8vJISkrCbj/+eZB6EUp2795NSkqK1WWIiIjICdixYwfJycnHXa9ehJKwsDDA/KHCw8MtrkZERESqw+VykZKSUvl3/HjqRSg5eMkmPDxcoURERKSeqW7XC3V0FREREZ+gUCIiIiI+QaFEREREfIJCiYiIiPgEhRIRERHxCQolIiIi4hMUSkRERMQnKJSIiIiIT6hxKPn1118ZOHAgSUlJ2Gw2vvrqq+N+ZtasWZx22mk4nU5OOeUU3n333RMoVURERBqyGoeSgoICunbtyquvvlqt9bdu3coll1zCOeecw7Jly7jnnnu45ZZb+PHHH2tcrIiIiDRcNR5m/qKLLuKiiy6q9vqTJk2iZcuWPP/88wC0b9+eOXPm8OKLLzJgwICa7l5EREQaqDrvUzJv3jz69+9fZdmAAQOYN29eXe9aRERE6pE6fyBfRkYG8fHxVZbFx8fjcrkoKioiKCjosM+UlJRQUlJS+drlctV1mSIiIvWLYUB5CZQXQVnFVF5sTmUV8/IScJdUrFcC7tI/LSs1Xx+c+twNkSmW/kg++ZTgCRMm8Nhjj1ldhoiISO0xDCgtgOJcKHFVzPPMdknen6Z8KM0z1y0thNJ8s11WaL4uOzgVAUbt1df5moYfShISEsjMzKyyLDMzk/Dw8COeJQEYO3Yso0ePrnztcrlISbH2QImIiFTyeKBoPxRkQUE2FGab86IDULgPCveb7xcdgKIcKM4x54a7buqxOcA/CPwCK+ZO8Ds4DwS/AHA4D587AiraARAWf/z91LE6DyVpaWl8//33VZbNmDGDtLS0o37G6XTidDrrujQREZGqDMMMFa5d4NptzvMyDk35GZCfZYaREw0Ydj8IjABnOASGm3NnGASEgjO0Yl7xOiDk0OQffGjuH/SneRA4/Gv3OFikxqEkPz+fTZs2Vb7eunUry5YtIzo6mmbNmjF27Fh27drF+++/D8Dtt9/OK6+8wgMPPMBNN93Ezz//zGeffcbUqVNr76cQERGpjoOh48C2imkr5KRXTDsgd6fZ36K6AiMhJBaCYyvm0RAcA0HRZjsoylwnKPLQ3D8YbLY6+OHqvxqHkkWLFnHOOedUvj54mWXYsGG8++677Nmzh/T09Mr3W7ZsydSpUxk1ahQvvfQSycnJ/Pvf/9btwCIiUnfKiiB7I2Sth30bYd+mimmL2V/jeELiIKIphCVBWAKEJZqXN0ITILQJhMabQcQvoO5/lkbEZhhGLfaSqRsul4uIiAhyc3MJDw+3uhwREfEVHjfs2wyZqyBzNexdY04HtnP0TqA2CE+CqBbmFNkMIlLMeWSKGUQUNmpFTf9+++TdNyIiIofxeCB7PexaDHuWw+5lZhgpKzzy+kFR0KQdxJwCsW3MeXRrM4j4B3qzcqkmhRIREfFNxbmwYwGk/wE7F8KuJUe+9OIfDHEdIL4jxHeCuPZmGAmJVd+NekahREREfENRDmybA9t+g+2/m5djDE/VdfxDIKk7JHWDxG6Q2BViWoPdYUHBUtsUSkRExBrucvMMyKYZsGUW7F56eAiJagnN+0ByL3Nq0g4c+tPVUOm/rIiIeE/hftjwI2z8ETb/bF6i+bOYU6DlWdCiHzRLg/BEa+oUSyiUiIhI3crdBeu+M6dtv1cddCwoCk7pD63PNcNIRFPr6hTLKZSIiEjtK8iGNV/Bqi9h+1yq3J4b3wnaXgRtBkDT09QfRCoplIiISO0oL4UNP8DSj2DTT1XPiKSkQvuB0O4SiG5lXY3i0xRKRETk5GSugcXvwsrPzYfQHZTYDTpdCZ2ugIhkq6qTekShREREaq68FNZ+AwvfgvS5h5aHJULXa6HbEHPAMpEaUCgREZHqK9gHi96CBZOhYK+5zOaAdhfDacPMDqvqIyInSKFERESOb99m+OM1s79IeZG5LDQBetwIPYaZz5IROUkKJSIicnRZ6+HXZ2HVfw4NbJbQBfrcDR0vA4e/peVJw6JQIiIih9u7Dn59xryl9+DtvKecD33vhhZn6JkyUicUSkRE5JCcHTBrAiz7mMow0vYSOOsB83kzInVIoURERMzh3+e8CPPfAHeJuazdpXDWg5DYxdrapNFQKBERacw8blj8Dvz8Lyg6YC5r3hfO/yck97S2Nml0FEpERBqr7fPgh/shY6X5Oq4D9H8M2pyvPiNiCYUSEZHGpmAf/PgQrJhivg6MgHMegZ43gUN/FsQ6+tcnItJYGAas/AKmPQiF+wCbOcbIueMhJMbq6kQUSkREGoWcHfDdKNg0w3wd1xEGvQzJPaytS+RPFEpERBoyw4AVn8L390OJCxxO8/beviM18Jn4HIUSEZGGqnC/eXZkzVfm6+TecNlrelCe+CyFEhGRhmjLLPjyb5CfAXY/OHsM9B2ljqzi0/SvU0SkIfG4zWfVzHoKMCD2VLjiTUjqbnVlIselUCIi0lDk74UvbzXPkgCcNhQufBoCgi0tS6S6FEpERBqC9Pnw2VDzco1/MFz6InS91uqqRGpEoUREpL5b8oHZodVTBk3awdXvQVw7q6sSqTGFEhGR+spdDtMfgfmvm6/bD4LLXgdnqLV1iZwghRIRkfqoKAc+vxG2/GK+PnssnPkA2O1WViX1gGEYlLo9FJW6KSpzU1jqpqjUTesmoQQFOCytTaFERKS+yd0JH14FWWvN/iOXvwEdBlldldQxwzAoLvOQU1RKblEZrqJyXEVl5JWY7fyScvKKy8krLiO/pJyCkvKKuZuC0nIKD85L3bg9xmHb/+bOvnRJjvT+D/YnCiUiIvVJ5mozkOTthtAEGPI5JHaxuio5AR6PwYHCUrLzS8nOLyE7v4R9+aXsLyhlX0EpBwpKOVB4cCojt7CMUrenVmvwd9gI9HcQHODgCDnF6xRKRETqi62/wpQh5nDxsW3hr/+ByBSrq5IjKC5zszuniIzcYvbkFrMnt4g9ucVkukrIyjPn2fkllJ9AEnDYbUQE+RMR5E94oB9hgf6EBfoRHuhPaKAfYYF+hDrNKdjpR6jTQXBAxesAsx0UYAYRf4dvXe5TKBERqQ/WfgdfDAd3KTTrA9d9DEFRVlfVaHk8BhmuYrbtKyB9XyHp+81px4Eidh0oIju/pNrbigz2JzbUSWxoADEhTmJCA4gOMaeoYHOKDPavmAIICXBgs9nq8KezjkKJiIivW/kFfHkbGG5oPxCu+Df4B1pdVaNQVOpmc1Y+m/aa0+asfLZmF7A1u4CS8mNfSgnyd5AUGUhSZBAJ4YEkRgQSFx5IfHggcWFO4sKdxIQ4CfDzrbMVVlIoERHxZUveh2/uBgzoeh0MekXPr6kDhmGw80ARq3fnsnZPHusz8liX4WL7/kKMo1xh8bPbSIkOptmfppToIJpGBtM0KoioYP8Ge0ajruhftoiIr5r/BvzwgNnueRNc/Lxu+a0le3KLWJaew7IdOazancuqXS5yi8qOuG5UsD9t4sI4JT6U1k1CadUkhJYxISRHBeHnY30y6juFEhERX7Rg8qFAknYnXPAv0P91n5Byt4e1e/JYsG0/i7btZ2l6Dhmu4sPW83fYODU+jPaJ4bRLMOdtE8KIDXVaUHXjpFAiIuJrFr8H399ntvuNgvMeVSCpAbfHYM1uF79vzmbu5n0s3rafglJ3lXXsNmibEE63lEi6JkfQqWkEp8aHqX+HxRRKRER8yfIp8O1Is512pwJJNWXkFjN7w15mrc9i7uZ9h12KCQv0o2fzKHq1jKZHsyg6J0cQHKA/gb5G/0VERHzFqi/hqzsAA3rdqks2x2AYBit35TJ9dSY/rc1kXUZelffDnH6ktoomrXUsaa1iaJsQhsOuY+nrFEpERHzBppkVt/164LShcNEzCiT/w+MxWLhtP9+v3MP0NZnsyT3UL8Rmgy7JkZx9ahPOatuELk0j1Am1HlIoERGx2q4l8OkN4CmDTlfCpS/pLpsKhmGwfGcu3y7fzdQVe6p0UA0OcHB22yac3yGeM9s0IUYdUus9hRIRESvt2wwfXQ1lBdDqbLhskgIJ5i27Xy7ZxX8W72RLdkHl8rBAPy7smMCFnRLoe0osgf7WPtVWapdCiYiIVfIy4YPLoTAbErvC4A/BL8DqqixT7vbw09pMPpqfzpxN2ZWDlgX5Ozi/QzwDuyZx5qmxOP0URBoqhRIRESuUFsDHV0POdohqCUO+AGeY1VVZItNVzCcL0pmyYEeVyzO9W0ZzVY9kLu6cSKhTf64aA/1XFhHxNo8H/vs32LMcgmPghi8hNM7qqrxu1a5c/v3bFr5bsafyabkxIQFc0yuFa3ul0DwmxOIKxdsUSkREvO3nx2Htt+AIgMEfQXQrqyvyGsMwmLUhi8m/bmHu5n2Vy3u1iOKvpzfnwk4JujzTiCmUiIh407KPYc4LZnvQy9A8zdp6vMTjMZixNpNXft7Eyl25ADjsNi7tksgt/VrROTnC4grFFyiUiIh4y/Z5FU/8Bc64D7pea209XmAYBj+uzmDiTxsrBzgLDnBwfe9m3NSvJUmRQRZXKL5EoURExBtcu+GzirFIOvwFznnY6orq3NxN2Tw9bR3Ld5pnRkKdfgzr05yb+7UiOqTx3mUkR6dQIiJS18pL4LOhUJAF8Z0a/Fgka3a7mPDDWn7bmA2YZ0Zu7teSm/u1JDJYYUSOTqFERKSuTRsDOxdCYIQ5FklAsNUV1Yl9+SU8P2MDUxak4zHA32FjSGpzRpxzCk3CNNqqHJ9CiYhIXVryASx6G7DBlW9BdEurK6p1ZW4PH8zbzsSfNuAqLgfg0i6JPDCgHc1iGmYAk7qhUCIiUld2L4Wp95rtcx6GNudbW08dWLYjhzH/WVHZibVDYjj/GNSR3i2jLa5M6iOFEhGRulCcC5/fCO4SaHsxnHGv1RXVqrziMp6fvoH35m3DMCAq2J/7B7RjcK8UHHY93VhOjEKJiEhtMwz49h44sA0imsFlrzeojq2/rN/L2P+srBwS/orTmvLIJR10R42ctBP6LXn11Vdp0aIFgYGBpKamsmDBgmOuP3HiRNq2bUtQUBApKSmMGjWK4uLiY35GRKTeWvIerP4S7H5w1dsQFGl1RbUiv6ScsV+uYPg7C8lwFdMsOpgPb07lhWu6KZBIrajxmZJPP/2U0aNHM2nSJFJTU5k4cSIDBgxg/fr1xMUd/uyGjz/+mDFjxvD222/Tp08fNmzYwI033ojNZuOFF16olR9CRMRnZK6GHx402+eNh5Re1tZTSxZs3c+9ny9jx/4iAG7q25L7B7QlKEBDwkvtsRnGwYdDV09qaiq9evXilVdeAcDj8ZCSksJdd93FmDFjDlv/zjvvZO3atcycObNy2b333sv8+fOZM2dOtfbpcrmIiIggNzeX8PDwmpQrIuI9pQXw5jmQvR5OOR+u/6zeX7Ypd3t4aeZGXvllE4YBTSODeO7qrqS1jrG6NKkHavr3u0a/LaWlpSxevJj+/fsf2oDdTv/+/Zk3b94RP9OnTx8WL15ceYlny5YtfP/991x88cVH3U9JSQkul6vKJCLi86aPMwNJWCJcXv8HSNuTW8T1k+fz8s9mILmmZzLT7jlDgUTqTI0u32RnZ+N2u4mPj6+yPD4+nnXr1h3xM9dffz3Z2dn069cPwzAoLy/n9ttv56GHHjrqfiZMmMBjjz1Wk9JERKy1YTosestsXz4JQmKtreck/bwuk3s/W86BwjJCnX5MuKIzA7smWV2WNHB1HuNnzZrFk08+yWuvvcaSJUv48ssvmTp1Ko8//vhRPzN27Fhyc3Mrpx07dtR1mSIiJ64gG74eYbZP/zu0OtvSck6Gx2PwwvT13PTuIg4UltG5aQTf3dVPgUS8okZnSmJjY3E4HGRmZlZZnpmZSUJCwhE/M27cOG644QZuueUWADp37kxBQQG33XYbDz/8MPYjnN50Op04nRqSWETqAcOAb0dCwV5o0g7Oe9Tqik6Yq7iMUVOWMXPdXgCGpTXnoUva4/RTZ1bxjhqdKQkICKBHjx5VOq16PB5mzpxJWlraET9TWFh4WPBwOMx/4DXsYysi4nuWfQTrvgO7P1wxGfwDra7ohGzam8dlr/zOzHV7CfCz88I1XXnsL50USMSranxL8OjRoxk2bBg9e/akd+/eTJw4kYKCAoYPHw7A0KFDadq0KRMmTABg4MCBvPDCC3Tv3p3U1FQ2bdrEuHHjGDhwYGU4ERGpl3LSD93+e+7DkNjF2npO0G8bs/j7h0vIKyknKSKQN27oSefkCKvLkkaoxqFk8ODBZGVlMX78eDIyMujWrRvTpk2r7Pyanp5e5czII488gs1m45FHHmHXrl00adKEgQMH8sQTT9TeTyEi4m2GAd/cDaX5kHI69Lnb6opOyJQF6Tz81SrcHoPeLaJ57a+nERuqy+dijRqPU2IFjVMiIj5n8Xvw7d3gFwh3zIWY1lZXVCMej8Fz09fz2qzNAFzevSlPXdlZl2ukVtX077eefSMiUlO5u2D6I2b73EfqXSApLfdw3+fL+Wb5bgDuPq8No/q3wWbTg/TEWgolIiI1YRjw3T1Q4oKmPc1bgOuRwtJy7vhwCbM3ZOHvsDHhii5c1SPZ6rJEAIUSEZGaWT4FNk4HRwD85VWw15/LHbmFZdz03kIWbz9AkL+DSTf04KxTm1hdlkglhRIRkerKz4JpFc/4OnsMxLWztp4a2JtXzNC3FrAuI4/wQD/eGd6LHs2jrS5LpAqFEhGR6pr+MBTnQEJn6DPS6mqqLSO3mOsm/8HW7AKahDl5/6betE/UTQPiexRKRESqY/MvsOJTwAYDXwJH/fj6/HMgaRoZxMe3ptI8JsTqskSOqH78VomIWKmsCKaONtu9b4OmPaytp5r25BZx3Zt/sG1fIU0jg5hy2+mkRAdbXZbIUdXv52qLiHjDb8/D/i0QlmjeAlwP/DmQJEcpkEj9oDMlIiLHsncdzJloti96GgJ9vy/GvvwShvx7fpVAkhylQCK+T2dKRESOxjDMyzaeMjj1Qmg/yOqKjstVXMbQtxewJauApIhABRKpVxRKRESOZuUXsP138AuCi58FHx/xtKjUzc3vLmT1bhcxIQF8eEuqAonUKwolIiJHUpJ3aCj5M++FyGbW1nMcpeUebv9wMQu3HSAs0I/3b+5NqyahVpclUiMKJSIiRzL7GcjPgKiWkHaX1dUck8dj8MAXy5m9IYsgfwfvDu9Fx6QIq8sSqTGFEhGR/5W1Af54zWxf9DT4B1pbz3E8O309Xy3bjZ/dxhs39NBIrVJvKZSIiPyZYcAP94On3OzceuoAqys6pg/+2M7rszYD8NSVXThTz7KRekyhRETkz9Z+C1tmgcMJF06wuppjmrEmk0e/XgXA6PNP1dN+pd5TKBEROais2Hy+DUDfuyG6lbX1HMOKnTnc9ckSPAZc2yuFu849xeqSRE6aQomIyEF/vAY56RCWBP1GWV3NUWW6irn1/UUUl3k469QmPH5ZJ2w+fruySHUolIiIAOTvhd9eMNv9H4UA33xoXXGZm9veX0Smq4Q2caG8cn13/B36KpeGQf+SRUQAfv4XlOZBUnfofI3V1RyRYRg88MUKlu/MJTLYn38P60lYoL/VZYnUGoUSEZGMVbD0A7M9YALYffOr8bVZm/lmuXnr72tDTqN5jG+ezRE5Ub75myci4i2GAT8+BIYHOl4OzdOsruiIflm3l+emrwfgsb90pE/rWIsrEql9CiUi0rhtmAZbZ5u3APd/zOpqjih9XyEjpyzFMOCG05szJLW51SWJ1AmFEhFpvNzlMGO82T79DojyvT/2RaVu/vbhYlzF5XRvFsm4SztYXZJInVEoEZHGa9mHkL0BgqLhjNFWV3MYwzB4+L8rWbvHRWxoAK8NOY0AP31tS8Olf90i0jiVFsAvFSO2nvUABPreA+w+nJ/Ol0t34bDbePm600iMCLK6JJE6pVAiIo3TvNfMpwBHNoeeN1ldzWFW7Mzhn9+uBmDMhe1Iax1jcUUidU+hREQan4Js+P0ls33eePBzWlvP/3AVl3Hnx0spcxsM6BjPLWe0tLokEa9QKBGRxmf2M+ZAaYndoOMVVldThWEYjP1yJen7C2kaGcQzV3bVEPLSaCiUiEjjsn8LLHrbbJ//mM8NlPbJgh1MXbEHP7uNl6/vTkSwRmyVxsO3fhtFROraLxPAUwatz4NWZ1tdTRVr97h4rKIfyf0D2nJasyiLKxLxLoUSEWk8MlfDys/N9nnjra3lfxSVurnrk6WUlHs4u20Tbj2jldUliXidQomINB4/PwEY0OEySOpmcTFVTfhhLZv25tMkzMnzV3fFblc/Eml8FEpEpHHYuQjWTwWbHc552Opqqvhl3V7en7cdgOeu7kpMqG/dDSTiLQolItI4zPynOe96PTQ51dpa/iQ7v4T7v1gBwI19WnDWqU0srkjEOgolItLwbZllPnTP7g9nP2h1NZUMw2DMf1aSnV/CqfGhjLmondUliVhKoUREGjbDgJmPm+2eN0FkM2vr+ZNPFuzgp7WZBDjsTBzcnUB/h9UliVhKoUREGrYN02DXIvAPhjPvs7qaSjv2F/KvqWsA8/bfDknhFlckYj2FEhFpuAwDfnnSbPe+DULjrK2ngsdjcN/nyyksddO7RTQ399Mw8iKgUCIiDdm6qZCxAgJCoe9Iq6up9P68bczfup8gfwfPXt1Ft/+KVFAoEZGGyeOBWRPMdurtEBxtbT0VtmYX8NS0dQA8dHE7mseEWFyRiO9QKBGRhmndt5C5CpzhkDbC6moAcHsM7v98OcVlHvq0jmFIanOrSxLxKQolItLweDzmM24ATr/DZ86SvPP7VhZtP0BIgINnrtJlG5H/pVAiIg3Pmv9C1lpwRsDpf7e6GgC27yvguenrAXj4kg4kRwVbXJGI71EoEZGGxeOGWU+b7bQREBRpaTlgDpI29suVFJd5SGsVw3W9U6wuScQnKZSISMOy5ivIXg+BEXD67VZXA8Bni3Ywd/M+Av3tPHVlZ2w2XbYRORKFEhFpODwemP2s2T59hBlMLJbpKuZfU9cCcO/5bXW3jcgxKJSISMOx7tuKviThkPo3q6vBMAzGfbWKvOJyuiZHMLxvC6tLEvFpCiUi0jAYxqGzJKm3+0RfkmmrMpi+JhM/u42nr+qCn0NfuSLHot8QEWkY1v8AmSvN0VtPv8PqanAVl/HoN6sBuOPs1rRL0LNtRI5HoURE6j/DgNkVd9z0vs0nxiV57sf17M0roWVsCCPOOcXqckTqBYUSEan/Ns6APcvAPwTS7rS6GpamH+CDP7YD8MRlnQj0d1hckUj9oFAiIvXbn8+S9LoZQmIsLafM7WHslysxDLjitKb0OSXW0npE6hOFEhGp37bMgl2LwC8I+txldTW88/tW1mXkERnsz8MXt7e6HJF6RaFEROq335435z2GQWicpaXsPFDIizM2AvDQxe2JCXVaWo9IfaNQIiL1V/p82PYb2P194izJY9+uoajMTe+W0VzdI9nqckTqHYUSEam/fnvOnHe9FiKsDQE/r8tkRsWYJP+6rJOGkhc5AScUSl599VVatGhBYGAgqampLFiw4Jjr5+TkMGLECBITE3E6nZx66ql8//33J1SwiAgAe1bAxulgs0O/UZaWUlzmrhyT5OZ+LTk1PszSekTqK7+afuDTTz9l9OjRTJo0idTUVCZOnMiAAQNYv349cXGHX88tLS3l/PPPJy4uji+++IKmTZuyfft2IiMja6N+EWmsDvYl6XgFxLS2tJTXZm1mx/4iEiMCufu8NpbWIlKf1TiUvPDCC9x6660MHz4cgEmTJjF16lTefvttxowZc9j6b7/9Nvv372fu3Ln4+/sD0KJFi5OrWkQat6wNsOZrs33GaEtL2ZZdwKTZmwEYd2kHQpw1/loVkQo1unxTWlrK4sWL6d+//6EN2O3079+fefPmHfEz33zzDWlpaYwYMYL4+Hg6derEk08+idvtPup+SkpKcLlcVSYRkUpzXgQMaHsxxHe0rAzDMHj0m9WUlns4o00sF3VKsKwWkYagRqEkOzsbt9tNfHx8leXx8fFkZGQc8TNbtmzhiy++wO128/333zNu3Dief/55/vWvfx11PxMmTCAiIqJySklJqUmZItKQ5eyAlZ+Z7TPutbSU6Wsymb0hiwCHnX/+RZ1bRU5Wnd994/F4iIuL480336RHjx4MHjyYhx9+mEmTJh31M2PHjiU3N7dy2rFjR12XKSL1xbxXwFMOLc+E5J6WlVFc5ubx79YAcNuZrWgZG2JZLSINRY0ufsbGxuJwOMjMzKyyPDMzk4SEI5+2TExMxN/fH4fj0LMf2rdvT0ZGBqWlpQQEBBz2GafTidOpQYdE5H8U7IPF75ntftb2JZk0ezM7DxSRFBHI38+xtqOtSENRozMlAQEB9OjRg5kzZ1Yu83g8zJw5k7S0tCN+pm/fvmzatAmPx1O5bMOGDSQmJh4xkIiIHNX8SVBeBIndoNXZlpWxY38hr88yO7c+dEl7ggPUuVWkNtT48s3o0aOZPHky7733HmvXruWOO+6goKCg8m6coUOHMnbs2Mr177jjDvbv38/IkSPZsGEDU6dO5cknn2TEiBG191OISMNXkgcL3jTb/UaBhf03npi6lpJyD2mtYrikc6JldYg0NDWO94MHDyYrK4vx48eTkZFBt27dmDZtWmXn1/T0dOz2Q1knJSWFH3/8kVGjRtGlSxeaNm3KyJEjefDBB2vvpxCRhm/xe1CcAzGnQPuBlpXx28Yspq3OwGG38Y9BHdW5VaQW2QzDMKwu4nhcLhcRERHk5uYSHh5udTki4m3lJfBSV8jbA4NehtOGWlJGmdvDhRN/ZXNWAcP7tuDRgdbdjixSH9T077eefSMivm/Fp2YgCUuCLoMtK+P9edvZnFVATEgA9/Q/1bI6RBoqhRIR8W0eD/z+ktlO+zv4WXNn3r78Eib+tAGA+we0JSLI35I6RBoyhRIR8W3rp8K+TRAYAT1utKyM56ZvIK+4nI5J4VzdUwM6itQFhRIR8V2GAXMmmu1et4LTmqfvrt6dy5SF6QA8OrAjDrs6t4rUBYUSEfFd2+fCrkXgcELq3ywpwTAMHvt2DYYBA7sm0btltCV1iDQGCiUi4rt+n2jOuw+B0DhLSpi6cg8Ltu4n0N/O2IvaWVKDSGOhUCIivilzNWycDjY7pN1pSQnFZW4mfL8OgDvOOoWkyCBL6hBpLBRKRMQ3/f5/5rz9IIix5tkyk3/dwq4c8/k2t53ZypIaRBoThRIR8T05O2DVF2a73z2WlJDpKua1iufbjLm4PUEBjuN8QkROlkKJiPieP14DTzm0PAuSultSwtPT1lFU5qZH8ygGdtHzbUS8QaFERHxL0QHzOTcAfe+2pITlO3L4cskuAMZf2kHPtxHxEoUSEfEtC9+CsgKI7wytz/P67g3D4J/frQHgitOa0jUl0us1iDRWCiUi4jvKimH+G2a7z11gwRmKb1fsYfH2AwT5O3hggG4BFvEmhRIR8R0rpkDBXghPhk5XeH33xWVunvp+LQB3nN2ahIhAr9cg0pgplIiIb/B4YO7LZjvt7+Dw/gPv/v3bFnbnFusWYBGLKJSIiG9Y//2hB++dNtTru9/7p1uAH7yoHYH+ugVYxNsUSkTEN8ytGCyt582WPHjv2R/XU1jqpnuzSAZ1TfL6/kVEoUREfEH6H7BjPjgCIPV2r+9+1a5cvliyE4BxugVYxDIKJSJivYNDyne9FsLivbprwzB4/DvzKcCDuiZxWrMor+5fRA5RKBERa2VvNPuTAKTd5fXd/7g6k/lb9+P0s/OgngIsYimFEhGx1tyXAQPaXgxNTvXqrkvK3TxZcQvwbWe2oqmeAixiKYUSEbFO/l5YPsVs9/H+kPLvzd1G+v5C4sKc3H6WNU8iFpFDFEpExDrz3wB3CST3gmane3XX+/JLeHnmJgDuG9CWEKefV/cvIodTKBERa5Tkw8J/m+0+d3t9SPkXf9pAXkk5HZPCueq0ZK/uW0SOTKFERKyx9EMozoHoVtDuEq/uekNmHh/PTwfMW4Dtdt0CLOILFEpExPvc5fDHq2Y77U6we3f01H9NXYvHgAEd4zm9VYxX9y0iR6dQIiLet/ZryEmH4Bjodr1Xd/3L+r38uiELf4eNhy5u79V9i8ixKZSIiHcZxqHB0nr/Dfy9dxtumdvDE1PNW4CH921J85gQr+1bRI5PoUREvGvbb7BnGfgFQa9bvLrrj+ens2lvPjEhAdx57ile3beIHJ9CiYh418GzJN2HQIj3+nPkFJby4k8bABh9wamEB/p7bd8iUj0KJSLiPZlrYNMMsNkhbYRXd/3SzI3kFJbRNj6MwT1TvLpvEakehRIR8Z55r5jz9gPNW4G9ZNPefD6Ytx2ARy5tj59DX30ivki/mSLiHa7dsOIzs91npFd3/eT3ayn3GPRvH8cZbZp4dd8iUn0KJSLiHfMngacMmveF5B5e2+3sDVn8vG4vfnbdAizi6xRKRKTuFbtg0Ttm24sP3itze3j8uzUADE1rQasmoV7bt4jUnEKJiNS9Je9BiQti20KbC7y224/+2M6mvflEhwQwsn8br+1XRE6MQomI1K3yUpj3mtnucxfYvfO1c6CglBd/2gjAvRecSkSQbgEW8XUKJSJSt1b9B/J2Q2gCdLnGa7t98acN5BaV0S4hjGt7NfPafkXkxCmUiEjdMQyYWzFY2um3g5/TK7tdn5HHh3+YtwCPH9gBh54CLFIvKJSISN3Z9BPsXQMBodBjuFd2aRgGj3+3Bo8BF3ZMoE/rWK/sV0ROnkKJiNSd318y5z1uhKBIr+xy+ppM5mzKJsBh1y3AIvWMQomI1I1di82H79n94PQ7vLLL4jJ35S3At53ZimYxwV7Zr4jUDoUSEakbBx+81+kqiEj2yi7f/HULOw8UkRAeyN/Pae2VfYpI7VEoEZHat28zrP3GbPe5yyu73JVTxGuzNgHw0CXtCQ7w88p+RaT2KJSISO2b9woYHjjlfEjo5JVdPjl1LcVlHnq3jGZgl0Sv7FNEapdCiYjUrvy9sPQjs93XOw/em7s5m6kr92C3wT8GdsRm0y3AIvWRQomI1K75b4C7BJr2gBb96nx35W4Pj31jdm4dktqcDknhdb5PEakbCiUiUntK8mDhZLPd9x7wwhmL9+ZtZ31mHpHB/ow+/9Q635+I1B2FEhGpPYvfg+JciDkF2l1S57vb6yrmxRkbAHjwwnZEhQTU+T5FpO4olIhI7SgvhT/+/OA9R53v8snv15JfUk7XlEgG90yp8/2JSN1SKBGR2rHqC3DtgtB46HJtne/ujy37+GrZbmw2ePwvHbHr+TYi9Z5CiYicPI/n0JDyp98B/oF1ursyt4fxX68C4PrezeiSHFmn+xMR71AoEZGTt2EaZK0DZzj0vKnOd/fe3G1syMwnKtif+we0rfP9iYh3KJSIyMkxDJjzgtnudTMERtTp7vbkFlXp3BoZrM6tIg2FQomInJztv8POheBwQmrdP3jvH9+spqDUzWnNIrlGnVtFGhSFEhE5OXNeNOfd/wph8XW6q5/WZPLj6kwcdhtPXN5ZnVtFGpgTCiWvvvoqLVq0IDAwkNTUVBYsWFCtz02ZMgWbzcZll112IrsVEV+zZzls+gls9jp/8F5haTmPfrMagFv6taR9okZuFWloahxKPv30U0aPHs2jjz7KkiVL6Nq1KwMGDGDv3r3H/Ny2bdu47777OOOMM064WBHxMXMmmvOOV0B0yzrd1Us/bWRXThFNI4MY2b9Nne5LRKxR41DywgsvcOuttzJ8+HA6dOjApEmTCA4O5u233z7qZ9xuN0OGDOGxxx6jVatWJ1WwiPiIfZthzVdmu9+oOt3V2j0u/j1nKwD//EtHggP86nR/ImKNGoWS0tJSFi9eTP/+/Q9twG6nf//+zJs376if++c//0lcXBw333xztfZTUlKCy+WqMomIj/n9JTA80OYCSOhUZ7txewzGfrkSt8fgwo4JnNe+bvutiIh1ahRKsrOzcbvdxMdX/VKIj48nIyPjiJ+ZM2cOb731FpMnT672fiZMmEBERETllJKiHvYiPsW1G5Z/Yrb7ja7TXb03dxvLduQQ6vTj0UEd6nRfImKtOr37Ji8vjxtuuIHJkycTGxtb7c+NHTuW3NzcymnHjh11WKWI1Njcl8FdCs37QvO0OtvNjv2FPDd9PQBjLmpHYkRQne1LRKxXowuzsbGxOBwOMjMzqyzPzMwkISHhsPU3b97Mtm3bGDhwYOUyj8dj7tjPj/Xr19O6devDPud0OnE6nTUpTUS8pSAbFr1jts+ou7MkhmHw8FerKCx107tlNNf3blZn+xIR31CjMyUBAQH06NGDmTNnVi7zeDzMnDmTtLTD/2+pXbt2rFy5kmXLllVOgwYN4pxzzmHZsmW6LCNSH/3xOpQXQWI3aH1ene3mv0t38euGLAL87Ey4QmOSiDQGNe7CPnr0aIYNG0bPnj3p3bs3EydOpKCggOHDhwMwdOhQmjZtyoQJEwgMDKRTp6od4CIjIwEOWy4i9UBxLiyo6B925n1gq5ugkJ1fwj+/WwPAyPPa0LpJaJ3sR0R8S41DyeDBg8nKymL8+PFkZGTQrVs3pk2bVtn5NT09HbtdA8WKNEgLJkNJLjRpD20vqbPdPPrNanIKy2ifGM5tZ2oYAZHGwmYYhmF1EcfjcrmIiIggNzeX8HCN4ihiidJCmNgJCvfBFZOhyzV1spvvVuzmzo+X4rDb+OrvfemcXLcP+BORulPTv986pSEi1bPkPTOQRLUwR3CtA1l5JYz7ahUAI85urUAi0sgolIjI8ZUVHxpSvt8ocNT+iKqGYfDIVys5UFhGu4Qw7jxXQ8mLNDYKJSJyfEs/gPwMCE+GrtfXyS6+Wb6bH1dn4me38fw1XQnw09eTSGOj33oRObbyEpjzotk+YxT4BdT6Lva6ihn/tfkE4LvObUPHJF22EWmMFEpE5NiWfQSuXRCWCN3+WuubNwyD+79YQW5RGR2Twvn7OYcPqCgijYNCiYgcXXkp/PaC2e57D/gH1vou3p+3ndkbsnD62Zk4uBv+Dn0tiTRW+u0XkaNbMQVyd0BIHPQYVuub35iZx5PfrwXgoYvb0yY+rNb3ISL1h0KJiByZuwx+fc5s9x0J/rX7MLyScjd3T1lGSbmHs05twtC05rW6fRGpfxRKROTIVnwGOdshOAZ6Dq/1zb8wfQNr97iIDgng2au7YKujIetFpP5QKBGRw7nL4NdnzHafuyEgpFY3P2djNm/+tgWAp6/sQlxY7fdVEZH6R6FERA63/BM4sA2CY6H3rbW66b2uYu75dCmGAdenNuP8DvG1un0Rqb8USkSkqvJS+PVZs91vVK2eJXF7DEZOWUZ2fintEsIYf2mHWtu2iNR/CiUiUtWyjyAn3bzjpudNtbrpl3/eyLwt+wgOcPDqkNMI9HfU6vZFpH5TKBGRQ8pL4LfnzfYZoyEguNY2PXdzNi/N3AjAE5d3onWT0Frbtog0DAolInLI0g/McUnCEqHHjbW22b2uYkZOWYZhwDU9k7m8e3KtbVtEGg6FEhExlRXDrwfPktxba+OSlJZ7+PtHS8jKK+HU+FD+MahjrWxXRBoehRIRMS1+B/J2Q3hTOG1orW32ialrWLT9AGFOPyb9tQfBAX61tm0RaVgUSkQESvIPjd561gPg56yVzf5n8U7em7cdgInXdqOV+pGIyDEolIgIzH8dCrMhuhV0G1Irm1y1K5eH/rsSgJHnteG89hqPRESOTaFEpLEr3A+/v2y2z3kYHP4nvcmsvBL+9sFiSso9nNsujpHntTnpbYpIw6dQItLYzf0/KMmFuI7Q8YqT3lxxmZvbPljErpwiWsaG8OLgbtjteq6NiByfQolIY5aXCX9MMtvnPgL2k/tKMAyD+79YwdL0HCKC/HlrWE8igk7+zIuINA4KJSKN2W/PQXkRNO0JbS866c29NHMj3y7fjZ/dxut/PU0dW0WkRhRKRBqrA9tg0Ttm+7zxYDu5SyxfL9vFxJ/MEVv/dVkn+rSOPckCRaSxUSgRaaxmPg6eMmh1NrQ666Q29fumbO77fDkAt53Zimt7N6uFAkWksVEoEWmMdi+FVV+Y7fP/eVKbWrUrl799sJgyt8HFnRN48MJ2tVCgiDRGCiUijY1hwIxHzXbnayCx6wlvavu+Am58ZwH5JeWktYrhxcHdcOhOGxE5QQolIo3N5pmwdTY4Asw7bk5QVl4JQ99eQHZ+KR0Sw3ljaA+cfo5aLFREGhuFEpHGxOOBGf8w271uhajmJ7SZAwWl3PDWfLbvKyQlOoh3b+pFeKBu/RWRk6NQItKYrPwMMleCMwLOvO+ENpFbWMZf35rPuow8moQ5ef+mVOLCAmu5UBFpjBRKRBqLsiL4+V9m+4xREBxd403kFZcx9J0FrN7tIiYkgI9vSaVlbEgtFyoijZVCiUhjMe8VyN0B4cmQenuNP15QUs6N7yxk+Y4cooL9+ejWVNrEh9VBoSLSWPlZXYCIeEFeBvz2otnu/w/wD6rRx3MLy7jx3QUsTc8hPNCPD25OpV1CeO3XKSKNmkKJSGPw8+NQVgDJvaDzVTX66L78Em54awFr9riICPLn/Zt606lpRB0VKiKNmUKJSEO3Zzks/chsD5hQo+HkM3KLGfLvP9icVUBsaAAf3JxK+0SdIRGRuqFQItKQGQb8+DBgQKerIKVXtT+6LbuAG96ez479RSRGBPLhLam01gP2RKQOKZSINGTrpsK238Av0OxLUk1L0g9wy3uL2F9QSvOYYD68OZWU6OC6q1NEBIUSkYarrBimP2y20+6EyJRqfWzaqgxGTllKSbmHTk3DefvGXhqHRES8QqFEpKGa+zIc2AZhidBvVLU+8s7vW/nnd2swDDi3XRwvX9edEKe+JkTEO/RtI9IQ5aTDb8+b7Qv+Bc5j9wUpKXfzj2/W8MmCdACGpDbjsUEd8XNoKCMR8R6FEpGG6MeHobwImveDTlcec9W9rmLu+GgJi7cfwGaDMRe247YzW2GrwV06IiK1QaFEpKHZ/DOs/QZsDrj4mWPeArw0/QC3f7iYTFcJ4YF+/N913Tm7bZwXixUROUShRKQhKS+F7x8w271vg/iOR1zNMAze/n0bT/2wljK3QZu4UN4c2lPPsRERSymUiDQkf7wG+zZCSBM4e8wRVzlQUMp9ny9n5rq9AFzUKYFnr+5KqDq0iojF9C0k0lAc2A6znzbb/R+DoMjDVpm/ZR8jpywjw1VMgJ+dcZe056+nN1f/ERHxCQolIg2BYcD390FZITTvC92ur/J2cZmb535cz1u/b8UwoFVsCC9f352OSXqGjYj4DoUSkYZgzdewcTrY/eHSiVU6ty5NP8C9ny9nS1YBANf0TObRgR01/oiI+Bx9K4nUd8W58MODZrvfKGhyKgCFpeW89NNGJv+2BY8BcWFOnrqyM+e2i7ewWBGRo1MoEanvfv4X5GdAdCs4414Apq/O4LFv17ArpwiAy7s35dGBHYgMDrCyUhGRY1IoEanPdi6GBZPN9qUvsiPPw2PfLuKntZkAJEcF8digjpzXXmdHRMT3KZSI1FflJfD1CMCgtOPVPL8unnd+n02p24O/w8atZ7TirnPbEBTgsLpSEZFqUSgRqa9+fRay1lIUEM3Fay5ka9EWAPqeEsM/BnakTXyYxQWKiNSMQolIPVS+cxn2317ADozOv4GtniBaNwnhkUs6cHbbJhp3RETqJYUSkXqk3O3hu2Xb6Tj1RtoYbqa6e7Mw+EweP+8Uru3dDH891VdE6jGFEpF6oKjUzWeLdjD5ty1c7vqIy/y3coAw9p/1BL+d2UP9RkSkQVAoEfFhe13FfDQ/nffnbeNAYRltbenc7fwKgKBBz3HDab2tLVBEpBYplIj4GMMwWLT9AO/N3ca0VRmUewwAWkf584nf2/jnlUPbiwnsPtjiSkVEatcJXYB+9dVXadGiBYGBgaSmprJgwYKjrjt58mTOOOMMoqKiiIqKon///sdcX6Sx2pdfwr9/28JFL/3G1ZPm8d2KPZR7DHo0j+Ll67ozo9uvROdtgOBYGPhSlaHkRUQaghqfKfn0008ZPXo0kyZNIjU1lYkTJzJgwADWr19PXFzcYevPmjWL6667jj59+hAYGMjTTz/NBRdcwOrVq2natGmt/BAi9VVJuZtZ67P4cslOZq7dW3lWJNDfzmXdmnJDWnPzoXnb5sB/XzE/NOhlCD38d01EpL6zGYZh1OQDqamp9OrVi1deMb8gPR4PKSkp3HXXXYwZM+a4n3e73URFRfHKK68wdOjQau3T5XIRERFBbm4u4eHhNSlXxOeUuT3M3byPb5fv5sfVGeQVl1e+1zU5gqt6pjCoaxIRQf7mwuJceL0v5O6A04aaoUREpB6o6d/vGp0pKS0tZfHixYwdO7Zymd1up3///sybN69a2ygsLKSsrIzo6OijrlNSUkJJSUnla5fLVZMyRXxOQUk5v27IYvqaTGauzcT1pyCSEB7IpV0SuapnMu0SjvBL+/0DZiCJagEDnvRe0SIiXlajUJKdnY3b7SY+vupzNOLj41m3bl21tvHggw+SlJRE//79j7rOhAkTeOyxx2pSmohPMQyDzVkFzN6QxewNWfyxZR+l5Z7K92NCAri4cyIDuybRs3kUdvtR+oes/AJWTAGbHS5/E5wapVVEGi6v3n3z1FNPMWXKFGbNmkVgYOBR1xs7diyjR4+ufO1yuUhJSfFGiSInbE9uEfM272Pe5n3M3byv8gm9BzWPCeaCDvFc0DGB05pF4ThaEDlo32b4dqTZPvN+aJZaR5WLiPiGGoWS2NhYHA4HmZmZVZZnZmaSkJBwzM8+99xzPPXUU/z000906dLlmOs6nU6cTmdNShPxKo/HYHNWPou2H2DRtgMs3r6fbfsKq6wT4LCT2iqas05twlmnNuGUuNDqD/9eXgKf3wil+dC8H5z1YO3/ECIiPqZGoSQgIIAePXowc+ZMLrvsMsDs6Dpz5kzuvPPOo37umWee4YknnuDHH3+kZ8+eJ1WwiLcZhsGe3GJW7cpl+c4cVuzMZcXOXHKLyqqsZ7dB56YRnN46hrRWMfRuGU1wwAmejJw+DjJWQHAMXDkZ7BqxVUQavhp/Y44ePZphw4bRs2dPevfuzcSJEykoKGD48OEADB06lKZNmzJhwgQAnn76acaPH8/HH39MixYtyMjIACA0NJTQ0NBa/FFETpxhGLiKytmZU8jOA0XsPFDE1ux81mfksT4jr0rH1IMC/e10TY6kZ4soejaPpkeLKMID/U++mLXfwoI3zPblb0B40slvU0SkHqhxKBk8eDBZWVmMHz+ejIwMunXrxrRp0yo7v6anp2O3HxqT7fXXX6e0tJSrrrqqynYeffRR/vGPf5xc9SLVUFruITu/hOz8EjJdJWS6itnrKibDVcye3Iopp4iCUvdRt+Fnt3FKXChdkyPpkhJB1+RI2iaE1f4D8PZvga9HmO0+d0Ob82t3+yIiPqzG45RYQeOUyEHlbg+u4nJyi8rIKSwlp2J+oMCc7ysoZX+BOd+XX0J2fulhl1mOJTY0gKZRwSRHBdEsOph2CWG0TQijVWwoAX51/ATe0gL49/mwdzUk94bh34OjFs68iIhYpE7HKRGpKY/HoKTcQ2FpOUVlbopK3RRWTuUUlLopKCmvmNwUlJaTV1xOfkk5+cVl5BWbr10V7fySwy+jVIef3UZMaADx4YEVk5O4sEASIwJJigwiMSKQxIgg6562axjwzd1mIAmJg2veUyARkUZHoaSeMQwDt8fAbRh4PFDu8eDxgNswKtvlHg9uj0G5x6DcbS6vOjcocx9aVnZwXm5Q5vFQVu6h1G0uL61ol5abU1lFu6TcQ0m525yXme3iP82LytwUl5nv14VQpx8RQf5EBPkTHRJAZLA/UcEBRIUEEBsaQHSIOTUJdRIb6iQiyP/oY4H4gj9eh1VfgN3PDCTqRyIijVCjDiVPT1vHlqx8PIb5P6pgYBhgYP7xN6DiPfMKl2GAxzi4joHHAA4u40/vGeZ7noq5UbHcfH2ofTBYGAfbhnlmwVMRPDwGfwog5tz3L7YdndPPTlCAg2B/B8FOP0ICHAQFOAh1+hEc4EeI049Qp4NQpz9hgX6EBvoR5vQjPMh8HR5oziOC/PGr7b4cVto2B6Y/YrYveAKa97G2HhERizTqUPLHln0sTc+xuoxaY7OZlykcdhsOmw0/h73ytZ/90Gs/hw0/ux1/x6FlAX52/B2HljkdFa/9bPg77AT4mcsC/Crafg6cfnac/nYCHA4C/e0E+pvLAv0PvQ70dxBUMT/uYGGN0YFt8NkwMNzQ+RpI/ZvVFYmIWKZRh5K/ndma7PwSbDawYauYg72icbBts1F1HZsNG+Yyh63qsoPrH9yGw2bDbrNht3Hotd18bbPZKt+32cBhP7SuzXYoXNjt5vb87OZnD26jMoD8aZnUI8W58PFgKMyGhC4w8CXzH5WISCPVqEPJhZ2OPQqtSJ1xl8MXN0HWOghLhOs/hYBgq6sSEbFUA7owL1KP/PgQbPoJ/ILguk/UsVVEBIUSEe+b/+ahEVuveBOSultbj4iIj1AoEfGmNV/DDw+Y7fPGQ4dB1tYjIuJDFEpEvGXrb/CfWwADegyHfqOtrkhExKcolIh4Q8ZKmHI9uEuh3aVwyfO600ZE5H8olIjUtQPb4MMrocQFzfvClW+B3aLh7EVEfJhCiUhdyt0F7w2C/EyI6wjXfgz+gVZXJSLikxRKROpKXga8NxBytkNUS/jrfyAo0uqqRER8lkKJSF3IzzLPkOzfDBHNYNi3EJ5odVUiIj5NoUSkthXuhw8ug+z1EJYEw76ByBSrqxIR8XmNeph5kVqXvxfe/wvsXQOh8eYZkuiWVlclIlIvKJSI1JbcXfD+INi3CUITYOjXEHuK1VWJiNQbCiUitWH/VjOQ5KRDRIoZSGJaW12ViEi9olAicrL2roUProC83RDdCoaqD4mIyIlQKBE5GdvmwCfXQ0kuNGlnniEJS7C6KhGRekmhRORErfoP/Pd2c+j4lNPhuk8gONrqqkRE6i2FEpGaMgyY9ypMf9h83X4gXDEZ/IOsrUtEpJ5TKBGpifJS+P5eWPK++Tr1dhjwpJ5lIyJSCxRKRKorPws+uwHS54HNDuc/Dmkj9LRfEZFaolAiUh17VsAn14FrJzgj4Kq3oM35VlclItKgKJSIHM/SD2HqvVBeDDGnwHVTILaN1VWJiDQ4CiUiR1NaAFPvg+Ufm69POR+unAxBUdbWJSLSQCmUiBxJ1nr4bBhkrTX7j5z7CPQdBXY9w1JEpK4olIj8mWHAwn/D9HFQXmQ+VO/Kt6DlGVZXJiLS4CmUiBzk2gNfj4DNM83Xrc6BK96E0Dhr6xIRaSQUSkQMA1Z/aXZmLToAfoFw/j+h1626XCMi4kUKJdK45e4yw8iGH8zXCV3M0Vnj2llbl4hII6RQIo2TxwOL34YZ/4DSPLD7wxn3mpNfgNXViYg0Sgol0vjsXAw/3A+7Fpuvk3vBoJchrr21dYmINHIKJdJ45GfBzH+Yg6EBBITBeeOh1816do2IiA9QKJGGr6wI5r8Bv70AJbnmsq7XQ/9HISzB2tpERKSSQok0XB43LPsYfnkS8nabyxK7wcXPQkpvS0sTEZHDKZRIw+PxwNqvYdbT5oisABEpcM7D0OUaXaoREfFRCiXScHjcsPq/8OuzkLXOXBYUBWfcB71uAf9Aa+sTEZFjUiiR+q+sGFZMgbmvwL6N5rLACDj975B6OwRFWlqeiIhUj0KJ1F+F+2HhW7DgDSjIMpcFRkLaCEj9mxlMRESk3lAokfpn12IzjKz6D5QXm8siUuD0O6D7DRAYbm19IiJyQhRKpH4odpn9RRa/A7uXHlqe2BX63A0d/gIOf+vqExGRk6ZQIr7L44Htc2DpR7DmaygvMpc7AqDjFWbn1eSeYLNZW6eIiNQKhRLxLYYBe5bByi/MMyOuXYfeiz0Vuv8Vug2BkFjLShQRkbqhUCLW83jMSzLrvoU138D+zYfec4ZDpyug2191VkREpIFTKBFrlBbCtjmw8UdY9/2hEVcB/IKg7YXQ6Uo45XyNLyIi0kgolIh3GIY5oNmWWbDpJzOQHLxzBsyH47U5H9pdAqdeCM5Qy0oVERFrKJRI3TAM2LcZ0ufC1l/NKT+z6joRKXBKf2h7MbQ6C/yc1tQqIiI+QaFEakdZMWSsgJ0LIf0PcyrYW3Udv0Bodjq0PhfaXABN2qmPiIiIVFIokZorLzUfdLdnuTntWgIZK8FTVnU9hxOa9oAWfaHlWZDcS/1DRETkqBRK5OgMA/L2mH1BMldXTKsgaz24Sw9fP6SJGTySe0KzPpDUXSFERESqTaFEoKwI9m+FfZsqps2Qvd4MHyWuI38mMMIcTTWxKyR2M8NIZDNdjhERkROmUNIYlJeag5Dl7oCcHeb8wHY4sM2c/nw77v+yOSC6JcR1gPhOEN/RnKJaKICIiEitUiipz9zlULjPvKslfy/kZ5iXW/IyzblrF7h2m+9hHHtbzgiIPQWiW0NMa2jSFmLbmm3dFSMiIl5wQqHk1Vdf5dlnnyUjI4OuXbvy8ssv07t376Ou//nnnzNu3Di2bdtGmzZtePrpp7n44otPuOgGyV0GxblQdACKcirm+6Fw/6F54T5zKsiGgiyzfbywcZDDCZEp5m24kSnmpZaolhVTCwiO1pkPERGxVI1Dyaeffsro0aOZNGkSqampTJw4kQEDBrB+/Xri4uIOW3/u3Llcd911TJgwgUsvvZSPP/6Yyy67jCVLltCpU6da+SG8zjDMEFFeZI5MWlYxlRZCWQGUHpzyoST/T+28isllzotdZrs41/z8ibDZITgWQuMgLAFCE8x5WAKEN4WIpuY8OEahQ0REfJrNMIxq/q+2KTU1lV69evHKK68A4PF4SElJ4a677mLMmDGHrT948GAKCgr47rvvKpedfvrpdOvWjUmTJlVrny6Xi4iICHJzcwkPD69Jucc27zU4sBXKS8zJXWL2vygvNu8uKS82x98o/9NUVmwGCMNde3X8mTMCgiIgMNI8exEUXTGPMsNHSMV0MIgEx4DdUTe1iIiInISa/v2u0ZmS0tJSFi9ezNixYyuX2e12+vfvz7x58474mXnz5jF69OgqywYMGMBXX3111P2UlJRQUlJS+drlOsodICdr9ZfmYF8nw2YH/xAICAb/IAgIhYAQ8A82584wc5kztGIebi47OAVFmssCI8y5Q918RESkcarRX8Ds7Gzcbjfx8fFVlsfHx7Nu3bojfiYjI+OI62dkZBx1PxMmTOCxxx6rSWknput10Opss7+FX8ChuV+Q2bmzcgoyx9vwq5j8KwKIfxA4AnRZREREpBb45P+Wjx07tsrZFZfLRUpKSu3vqNfNtb9NEREROSE1CiWxsbE4HA4yM6s+WC0zM5OEhIQjfiYhIaFG6wM4nU6cTt2GKiIi0pjYa7JyQEAAPXr0YObMmZXLPB4PM2fOJC0t7YifSUtLq7I+wIwZM466voiIiDRONb58M3r0aIYNG0bPnj3p3bs3EydOpKCggOHDhwMwdOhQmjZtyoQJEwAYOXIkZ511Fs8//zyXXHIJU6ZMYdGiRbz55pu1+5OIiIhIvVbjUDJ48GCysrIYP348GRkZdOvWjWnTplV2Zk1PT8duP3QCpk+fPnz88cc88sgjPPTQQ7Rp04avvvqq/o5RIiIiInWixuOUWKHOxikRERGROlPTv9816lMiIiIiUlcUSkRERMQnKJSIiIiIT1AoEREREZ+gUCIiIiI+QaFEREREfIJCiYiIiPgEhRIRERHxCT75lOD/dXB8N5fLZXElIiIiUl0H/25Xd5zWehFK8vLyAEhJSbG4EhEREampvLw8IiIijrtevRhm3uPxsHv3bsLCwrDZbLW2XZfLRUpKCjt27NDw9dWkY1ZzOmYnRset5nTMak7HrOZqcswMwyAvL4+kpKQqz8U7mnpxpsRut5OcnFxn2w8PD9c/xhrSMas5HbMTo+NWczpmNadjVnPVPWbVOUNykDq6ioiIiE9QKBERERGf0KhDidPp5NFHH8XpdFpdSr2hY1ZzOmYnRset5nTMak7HrObq8pjVi46uIiIi0vA16jMlIiIi4jsUSkRERMQnKJSIiIiIT1AoEREREZ/QqEPJq6++SosWLQgMDCQ1NZUFCxZYXZLPmDBhAr169SIsLIy4uDguu+wy1q9fX2Wd4uJiRowYQUxMDKGhoVx55ZVkZmZaVLFveeqpp7DZbNxzzz2Vy3S8jmzXrl389a9/JSYmhqCgIDp37syiRYsq3zcMg/Hjx5OYmEhQUBD9+/dn48aNFlZsLbfbzbhx42jZsiVBQUG0bt2axx9/vMqzRRr7Mfv1118ZOHAgSUlJ2Gw2vvrqqyrvV+f47N+/nyFDhhAeHk5kZCQ333wz+fn5XvwpvOtYx6ysrIwHH3yQzp07ExISQlJSEkOHDmX37t1VtlEbx6zRhpJPP/2U0aNH8+ijj7JkyRK6du3KgAED2Lt3r9Wl+YTZs2czYsQI/vjjD2bMmEFZWRkXXHABBQUFleuMGjWKb7/9ls8//5zZs2eze/durrjiCgur9g0LFy7kjTfeoEuXLlWW63gd7sCBA/Tt2xd/f39++OEH1qxZw/PPP09UVFTlOs888wz/93//x6RJk5g/fz4hISEMGDCA4uJiCyu3ztNPP83rr7/OK6+8wtq1a3n66ad55plnePnllyvXaezHrKCggK5du/Lqq68e8f3qHJ8hQ4awevVqZsyYwXfffcevv/7Kbbfd5q0fweuOdcwKCwtZsmQJ48aNY8mSJXz55ZesX7+eQYMGVVmvVo6Z0Uj17t3bGDFiROVrt9ttJCUlGRMmTLCwKt+1d+9eAzBmz55tGIZh5OTkGP7+/sbnn39euc7atWsNwJg3b55VZVouLy/PaNOmjTFjxgzjrLPOMkaOHGkYho7X0Tz44INGv379jvq+x+MxEhISjGeffbZyWU5OjuF0Oo1PPvnEGyX6nEsuucS46aabqiy74oorjCFDhhiGoWP2vwDjv//9b+Xr6hyfNWvWGICxcOHCynV++OEHw2azGbt27fJa7Vb532N2JAsWLDAAY/v27YZh1N4xa5RnSkpLS1m8eDH9+/evXGa32+nfvz/z5s2zsDLflZubC0B0dDQAixcvpqysrMoxbNeuHc2aNWvUx3DEiBFccsklVY4L6HgdzTfffEPPnj25+uqriYuLo3v37kyePLny/a1bt5KRkVHluEVERJCamtpoj1ufPn2YOXMmGzZsAGD58uXMmTOHiy66CNAxO57qHJ958+YRGRlJz549K9fp378/drud+fPne71mX5Sbm4vNZiMyMhKovWNWLx7IV9uys7Nxu93Ex8dXWR4fH8+6dessqsp3eTwe7rnnHvr27UunTp0AyMjIICAgoPIf5EHx8fFkZGRYUKX1pkyZwpIlS1i4cOFh7+l4HdmWLVt4/fXXGT16NA899BALFy7k7rvvJiAggGHDhlUemyP9rjbW4zZmzBhcLhft2rXD4XDgdrt54oknGDJkCICO2XFU5/hkZGQQFxdX5X0/Pz+io6N1DDH7xz344INcd911lQ/kq61j1ihDidTMiBEjWLVqFXPmzLG6FJ+1Y8cORo4cyYwZMwgMDLS6nHrD4/HQs2dPnnzySQC6d+/OqlWrmDRpEsOGDbO4Ot/02Wef8dFHH/Hxxx/TsWNHli1bxj333ENSUpKOmdS5srIyrrnmGgzD4PXXX6/17TfKyzexsbE4HI7D7nzIzMwkISHBoqp805133sl3333HL7/8QnJycuXyhIQESktLycnJqbJ+Yz2GixcvZu/evZx22mn4+fnh5+fH7Nmz+b//+z/8/PyIj4/X8TqCxMREOnToUGVZ+/btSU9PB6g8NvpdPeT+++9nzJgxXHvttXTu3JkbbriBUaNGMWHCBEDH7Hiqc3wSEhIOu+mhvLyc/fv3N+pjeDCQbN++nRkzZlSeJYHaO2aNMpQEBATQo0cPZs6cWbnM4/Ewc+ZM0tLSLKzMdxiGwZ133sl///tffv75Z1q2bFnl/R49euDv71/lGK5fv5709PRGeQzPO+88Vq5cybJlyyqnnj17MmTIkMq2jtfh+vbte9it5hs2bKB58+YAtGzZkoSEhCrHzeVyMX/+/EZ73AoLC7Hbq351OxwOPB4PoGN2PNU5PmlpaeTk5LB48eLKdX7++Wc8Hg+pqaler9kXHAwkGzdu5KeffiImJqbK+7V2zE6gY26DMGXKFMPpdBrvvvuusWbNGuO2224zIiMjjYyMDKtL8wl33HGHERERYcyaNcvYs2dP5VRYWFi5zu233240a9bM+Pnnn41FixYZaWlpRlpamoVV+5Y/331jGDpeR7JgwQLDz8/PeOKJJ4yNGzcaH330kREcHGx8+OGHles89dRTRmRkpPH1118bK1asMP7yl78YLVu2NIqKiiys3DrDhg0zmjZtanz33XfG1q1bjS+//NKIjY01Hnjggcp1Gvsxy8vLM5YuXWosXbrUAIwXXnjBWLp0aeWdItU5PhdeeKHRvXt3Y/78+cacOXOMNm3aGNddd51VP1KdO9YxKy0tNQYNGmQkJycby5Ytq/I3oaSkpHIbtXHMGm0oMQzDePnll41mzZoZAQEBRu/evY0//vjD6pJ8BnDE6Z133qlcp6ioyPj73/9uREVFGcHBwcbll19u7Nmzx7qifcz/hhIdryP79ttvjU6dOhlOp9No166d8eabb1Z53+PxGOPGjTPi4+MNp9NpnHfeecb69estqtZ6LpfLGDlypNGsWTMjMDDQaNWqlfHwww9X+ePQ2I/ZL7/8csTvr2HDhhmGUb3js2/fPuO6664zQkNDjfDwcGP48OFGXl6eBT+NdxzrmG3duvWofxN++eWXym3UxjGzGcafhgEUERERsUij7FMiIiIivkehRERERHyCQomIiIj4BIUSERER8QkKJSIiIuITFEpERETEJyiUiIiIiE9QKBERERGfoFAiIiIiPkGhRERERHyCQomIiIj4BIUSERER8Qn/D0AFvHwlu8uNAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.models import Model\n",
        "from keras.layers import Input\n",
        "from keras.layers import LSTM\n",
        "i1=Input(shape=(3,1))\n",
        "lstm1=LSTM(2)(i1)\n",
        "model=Model(inputs=i1,outputs=lstm1)\n",
        "data=np.array([0.1,0.2,0.3]).reshape((1,3,1))\n",
        "print(model.predict(data))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A2xNterxTcEv",
        "outputId": "0b2695db-37b5-42f6-bf86-2a8ee08a822d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 368ms/step\n",
            "[[0.05990469 0.04206322]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "9HL5nn1bSMhp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ELmAipuFTahm"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}