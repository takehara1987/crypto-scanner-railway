### CÓDIGO FINAL E COMPLETO (v45 - PARA RAILWAY COM 200 ATIVOS) ###

# ==============================================================================
# ETAPA 0: IMPORTAÇÕES E CONFIGURAÇÃO DA APLICAÇÃO
# ==============================================================================
from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from scipy.signal import find_peaks
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

btc_data_cache = None

# ==============================================================================
# DEFINIÇÃO DAS FUNÇÕES DE ANÁLISE
# ==============================================================================

def get_btc_data():
    global btc_data_cache
    if btc_data_cache is None:
        print("INFO: Buscando dados do Bitcoin para análise de correlação...")
        btc_data_cache = yf.Ticker("BTC-USD").history(period="1y")
        if not btc_data_cache.empty:
            btc_data_cache['MME21'] = ta.ema(btc_data_cache['Close'], length=21)
    return btc_data_cache

def buscar_gatilho_horario(ticker, data_sinal, tipo_setup):
    try:
        dados_h1 = yf.Ticker(ticker).history(period="5d", interval="1h")
        if dados_h1.empty: return None
        dados_h1['MME21'] = ta.ema(dados_h1['Close'], length=21)
        dados_sinal_h1 = dados_h1[dados_h1.index.date == data_sinal.date()]
        
        for i in range(1, len(dados_sinal_h1)):
            vela_anterior = dados_sinal_h1.iloc[i-1]; vela_atual = dados_sinal_h1.iloc[i]
            gatilho = False
            if 'COMPRA' in tipo_setup and vela_anterior['Close'] < vela_anterior['MME21'] and vela_atual['Close'] > vela_atual['MME21']:
                gatilho = True
            elif 'VENDA' in tipo_setup and vela_anterior['Close'] > vela_anterior['MME21'] and vela_atual['Close'] < vela_atual['MME21']:
                gatilho = True
            if gatilho:
                return {'gatilho_encontrado': True, 'preco_entrada': vela_atual['Close'], 'hora_entrada': vela_atual.name.strftime('%Y-%m-%d %H:%M')}
        return {'gatilho_encontrado': False}
    except Exception:
        return None

def analisar_ativo_mtf(ticker):
    try:
        dados_d1 = yf.Ticker(ticker).history(period="1y")
        if dados_d1.empty or len(dados_d1) < 201: return None
        
        # --- Cálculo de Indicadores ---
        dados_d1['MME200'] = ta.ema(dados_d1['Close'], length=200)
        dados_d1['Volume_MA20'] = dados_d1['Volume'].rolling(window=20).mean()
        dados_d1['RSI'] = ta.rsi(dados_d1['Close'], length=14)
        dados_d1['ATR'] = ta.atr(dados_d1['High'], dados_d1['Low'], dados_d1['Close'], length=14)
        bbands = ta.bbands(dados_d1['Close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            dados_d1['BB_Width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']
            dados_d1['BB_Width_MA20'] = dados_d1['BB_Width'].rolling(window=20).mean()
        else:
            dados_d1['BB_Width'], dados_d1['BB_Width_MA20'] = 0, 0
        
        dados_d1['range_low_30d'] = dados_d1['Low'].rolling(window=30).min()
        
        # --- Verificação de Setups no Penúltimo Dia ---
        penultimo_dia = dados_d1.iloc[-2]; antepenultimo_dia = dados_d1.iloc[-3]; ultimo_dia = dados_d1.iloc[-1]
        setups_encontrados = []
        score = 0
        
        # FILTROS GERAIS
        regime_nao_explosivo = penultimo_dia['BB_Width'] < penultimo_dia['BB_Width_MA20']
        tendencia_de_alta = penultimo_dia['Close'] > penultimo_dia['MME200']
        btc_em_alta = True
        if ticker != "BTC-USD":
            btc_data = get_btc_data()
            if btc_data is None or btc_data.empty: return None
            btc_no_dia = btc_data.loc[btc_data.index.asof(penultimo_dia.name)]
            btc_em_alta = btc_no_dia['Close'] > btc_no_dia['MME21']

        if tendencia_de_alta and btc_em_alta and regime_nao_explosivo:
            # Setup 1: Wyckoff Spring com Volume
            suporte_range = antepenultimo_dia['range_low_30d']
            if antepenultimo_dia['Low'] < suporte_range and penultimo_dia['Close'] > suporte_range and penultimo_dia['Volume'] > penultimo_dia['Volume_MA20']:
                score += 1
                setups_encontrados.append({'tipo': 'COMPRA_SPRING', 'stop_base': antepenultimo_dia['Low'], 'atr': penultimo_dia['ATR']})
        
        if not setups_encontrados: return None

        # --- Lógica de Classificação ---
        SCORE_MINIMO_SETUP = 2
        
        setup_principal = setups_encontrados[0]
        stop_dinamico = setup_principal['stop_base'] - (setup_principal['atr'] * 0.5)

        if score >= SCORE_MINIMO_SETUP:
            resultado_h1 = buscar_gatilho_horario(ticker, ultimo_dia.name, setup_principal['tipo'])
            if resultado_h1 and resultado_h1['gatilho_encontrado']:
                preco_entrada_h1 = resultado_h1['preco_entrada']; stop_loss = stop_dinamico
                risco = abs(preco_entrada_h1 - stop_loss)
                alvo = preco_entrada_h1 + (risco * 3)
                return {'status': 'CONFIRMADO', 'ativo': ticker, 'estrategia': f"{setup_principal['tipo']}_MTF (Score: {score})", 'hora_gatilho': resultado_h1['hora_entrada'], 'entrada': preco_entrada_h1, 'stop': stop_loss, 'alvo': alvo}
            else:
                return {'status': 'AGUARDANDO_GATILHO', 'ativo': ticker, 'estrategia': setup_principal['tipo'], 'data_setup': penultimo_dia.name.strftime('%Y-%m-%d'), 'stop_potencial': stop_dinamico, 'score': score}
        elif score == 1:
            return {'status': 'EM_OBSERVACAO', 'ativo': ticker, 'estrategia': setup_principal['tipo'], 'data_setup': penultimo_dia.name.strftime('%Y-%m-%d'), 'stop_potencial': stop_dinamico, 'score': score}
            
    except Exception:
        return None
    return None

# ==============================================================================
# O PONTO DE ENTRADA DA API (ENDPOINT)
# ==============================================================================
@app.route('/scan', methods=['GET'])
def scan_market():
    # Watchlist expandida para 200 ativos
    watchlist = [
        # Top Tier & Large Caps (50)
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "SHIB-USD", "DOT-USD",
        "LINK-USD", "TON-USD", "TRX-USD", "MATIC-USD", "BCH-USD", "LTC-USD", "NEAR-USD", "UNI-USD", "XLM-USD", "ATOM-USD",
        "ETC-USD", "XMR-USD", "ICP-USD", "HBAR-USD", "VET-USD", "FIL-USD", "APT-USD", "CRO-USD", "LDO-USD", "ARB-USD",
        "QNT-USD", "AAVE-USD", "ALGO-USD", "STX-USD", "FTM-USD", "EOS-USD", "SAND-USD", "MANA-USD", "THETA-USD", "AXS-USD",
        "RNDR-USD", "XTZ-USD", "SUI-USD", "PEPE-USD", "INJ-USD", "GALA-USD", "SNX-USD", "OP-USD", "KAS-USD", "TIA-USD",
        # Mid Caps (50)
        "MKR-USD", "RUNE-USD", "WIF-USD", "JUP-USD", "SEI-USD", "EGLD-USD", "FET-USD", "FLR-USD", "BONK-USD", "BGB-USD",
        "BEAM-USD", "DYDX-USD", "AGIX-USD", "NEO-USD", "WLD-USD", "ROSE-USD", "PYTH-USD", "GNO-USD", "CHZ-USD", "MINA-USD",
        "FLOW-USD", "KCS-USD", "FXS-USD", "KLAY-USD", "GMX-USD", "RON-USD", "CFX-USD", "CVX-USD", "ZEC-USD", "AIOZ-USD",
        "WEMIX-USD", "ENA-USD", "TWT-USD", "CAKE-USD", "CRV-USD", "FLOKI-USD", "BTT-USD", "1INCH-USD", "GMT-USD", "ZIL-USD",
        "ANKR-USD", "JASMY-USD", "KSM-USD", "LUNC-USD", "USTC-USD", "CELO-USD", "IOTA-USD", "HNT-USD", "RPL-USD", "FTT-USD",
        # Additional Mid/Small Caps (100)
        "XDC-USD", "PAXG-USD", "DASH-USD", "ENS-USD", "BAT-USD", "ZRX-USD", "YFI-USD", "SUSHI-USD", "UMA-USD", "REN-USD",
        "KNC-USD", "BAL-USD", "LRC-USD", "OCEAN-USD", "POWR-USD", "RLC-USD", "BAND-USD", "TRB-USD", "API3-USD", "BLZ-USD",
        "PERP-USD", "COTI-USD", "STORJ-USD", "SKL-USD", "CTSI-USD", "NKN-USD", "OGN-USD", "NMR-USD", "IOTX-USD", "AUDIO-USD",
        "CVC-USD", "LOOM-USD", "MDT-USD", "REQ-USD", "RLY-USD", "TRU-USD", "ACH-USD", "AGLD-USD", "ALCX-USD", "AMP-USD",
        "ARPA-USD", "AUCTION-USD", "BADGER-USD", "BICO-USD", "BNT-USD", "BOND-USD", "CLV-USD", "CTX-USD", "DDX-USD", "DIA-USD",
        "DREP-USD", "ELF-USD", "FARM-USD", "FORTH-USD", "GHST-USD", "GTC-USD", "HIGH-USD", "IDEX-USD", "KEEP-USD", "KP3R-USD",
        "LCX-USD", "MASK-USD", "MLN-USD", "NEST-USD", "NU-USD", "ORN-USD", "OXT-USD", "PLA-USD", "POLS-USD", "POND-USD",
        "RAI-USD", "RGT-USD", "SHPING-USD", "SPELL-USD", "SUPER-USD", "WNXM-USD", "YFII-USD", "RAD-USD", "COVAL-USD", "OMG-USD",
        "ENJ-USD", "WAVES-USD", "ICX-USD", "QTUM-USD", "ONT-USD", "IOST-USD", "DGB-USD", "SC-USD", "LSK-USD", "ARDR-USD",
        "SYS-USD", "STEEM-USD", "NEXO-USD", "HOT-USD", "BTG-USD", "ZEN-USD", "SRM-USD", "DCR-USD", "RVN-USD", "NANO-USD"
    ]
    watchlist = list(dict.fromkeys(watchlist))[:200]
    
    alertas_confirmados = []; setups_aguardando_gatilho = []; ativos_em_observacao = []
    
    get_btc_data()
    
    for ativo in watchlist:
        print(f"Analisando {ativo}...")
        resultado = analisar_ativo_mtf(ativo)
        if resultado:
            if resultado['status'] == 'CONFIRMADO':
                alertas_confirmados.append(resultado)
            elif resultado['status'] == 'AGUARDANDO_GATILHO':
                setups_aguardando_gatilho.append(resultado)
            elif resultado['status'] == 'EM_OBSERVACAO':
                ativos_em_observacao.append(resultado)

    return jsonify({
        'sinaisConfirmados': alertas_confirmados,
        'setupsEmAndamento': setups_aguardando_gatilho,
        'ativosEmObservacao': ativos_em_observacao
    })

@app.route('/')
def health_check():
    return "Servidor de análise v45 a funcionar!"

if __name__ == "__main__":
    # A Railway usa a variável de ambiente PORT para saber onde correr o servidor.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
