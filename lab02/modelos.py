import torch
import torch.nn as nn


class LSTMBase(nn.Module):
    """LSTM; el último estado oculto de la última capa predice las 48 horas siguientes.

    Por defecto usa la configuración de la Tabla S1 del paper (256 unidades, 2 capas, dropout 0.2).
    """

    def __init__(self, n_entradas=12, oculto=256, capas=2, dropout=0.2, horizonte=48):
        super().__init__()
        self.lstm = nn.LSTM(n_entradas, oculto, num_layers=capas, dropout=dropout, batch_first=True)
        self.salida = nn.Linear(oculto, horizonte)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.salida(h[-1])


class FDMLP(nn.Module):
    """MLP en el dominio de la frecuencia sobre el eje de variables (Ec. 11-15 del paper).

    Cada valor se proyecta a un vector de dimensión `embed`, se aplica la FFT a lo largo
    de las variables, un MLP complejo de dos capas con CReLU y la IFFT.
    Con frecuencia=False se omiten la FFT/IFFT (ablación).
    """

    def __init__(self, n_vars, embed=256, oculto=256, frecuencia=True):
        super().__init__()
        self.n_vars = n_vars
        self.frecuencia = frecuencia
        self.embedding = nn.Parameter(torch.randn(1, embed))
        # índice 0: parte real, índice 1: parte imaginaria
        self.w1 = nn.Parameter(0.02 * torch.randn(2, embed, oculto))
        self.b1 = nn.Parameter(0.02 * torch.randn(2, oculto))
        self.w2 = nn.Parameter(0.02 * torch.randn(2, oculto, embed))
        self.b2 = nn.Parameter(0.02 * torch.randn(2, embed))

    def capa_compleja(self, re, im, w, b):
        re_sal = re @ w[0] - im @ w[1] + b[0]
        im_sal = re @ w[1] + im @ w[0] + b[1]
        return re_sal, im_sal

    def forward(self, x):
        h = x.unsqueeze(-1) * self.embedding  # [B, T, n_vars, embed]

        if not self.frecuencia:
            z = torch.relu(h @ self.w1[0] + self.b1[0])
            z = z @ self.w2[0] + self.b2[0]
            return (z + h).flatten(2)

        espectro = torch.fft.rfft(h, dim=2)
        re, im = self.capa_compleja(espectro.real, espectro.imag, self.w1, self.b1)
        re, im = torch.relu(re), torch.relu(im)
        re, im = self.capa_compleja(re, im, self.w2, self.b2)
        z = torch.fft.irfft(torch.complex(re, im), n=self.n_vars, dim=2)
        return (z + h).flatten(2)  # [B, T, n_vars * embed]


class FDMLPLSTM(nn.Module):
    """FDMLP seguido de una LSTM, con la configuración de la Tabla S5 del paper."""

    def __init__(self, n_vars, embed=256, oculto_fdmlp=256, oculto_lstm=128, capas=2, frecuencia=True):
        super().__init__()
        self.fdmlp = FDMLP(n_vars, embed, oculto_fdmlp, frecuencia)
        self.lstm = LSTMBase(n_vars * embed, oculto_lstm, capas)

    def forward(self, x):
        return self.lstm(self.fdmlp(x))


MODELOS = {
    "lstm": lambda n_vars: LSTMBase(n_vars),
    "fdmlp": lambda n_vars: FDMLPLSTM(n_vars),
    "mlp": lambda n_vars: FDMLPLSTM(n_vars, frecuencia=False),
}
