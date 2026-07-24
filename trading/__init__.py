"""Unified backtest / shadow / paper / live trading framework.

See docs/architecture.md. The strategy code never knows which mode it runs
in; only the DataFeed and Broker adapters change between modes.
"""
