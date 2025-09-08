# src/utils/peer_discovery.py

def get_peers(ticker: str) -> list[str]:
    """
    Returns a list of key competitors for a given company ticker.
    This is a simplified implementation; a real-world version might use an API
    or a more sophisticated discovery method.
    """
    peer_map = {
        'AAPL': ['MSFT', 'GOOGL', 'QCOM', 'SNE'],
        'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'IBM'],
        'NVDA': ['AMD', 'INTC', 'QCOM'],
        'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'],
        'TSLA': ['F', 'GM', 'RIVN']
    }
    return peer_map.get(ticker, [])
