| key | source URL | licence | what the loader produces | synthetic parts | recipe |
|---|---|---|---|---|---|
| sec_suspensions | https://www.sec.gov/enforcement-litigation/trading-suspensions (+ `/rss`), order PDFs linked from it; daily prices from Yahoo Finance (yfinance) | Public domain (SEC list and orders); Yahoo prices under Yahoo terms, not redistributable | One daily `BarFrame` per suspended ticker: trading days in [suspension − 120 d, suspension + 20 d]; weak `Episode("pump", "real", t0 = suspension − 30 d, t1 = suspension day)`, label = bar date in [t0, t1]. Available features (`feat_mask`): ret (close-to-close log return), rv (log(high / low)), vol_trade (volume / median volume of the bars before the label window). `meta["survivorship_dropped"]` lists every list item / ticker that produced no frame, with the reason; `meta["survivorship_summary"]` gives the counts. | None. Tickers without Yahoo history are dropped and recorded, never synthesised. | Save list pages as `data_raw/sec_suspensions/list_p{N}.html`, the feed as `rss.xml`, orders as `orders/<name>.pdf` (or `.txt` from `pdftotext` when pypdf is absent), prices as `prices/<TICKER>.csv`; or `prepare(root, download=True, user_agent="Name email", pages=N)`. sec.gov refuses requests without a declared User-Agent. |

Ticker regexes (`extract_ticker`, case-sensitive, symbol = 2–5 capitals not followed by a letter, stop list for OTC / SEC / CIK / INC …):
- R1 `(ticker|symbol)s? [:,-]* (of|is|was|are|were)? ["“(]* SYMBOL` — e.g. *under the ticker symbol "ABCD"*
- R2 `(OTC|OTCQB|OTCQX|OTCBB [Pink|Link|Markets] | Pink Sheets | NASDAQ | NYSE [American|Arca] | AMEX) : SYMBOL` — e.g. *(OTC Pink: ABCD)*
- R3 further quoted symbols within 80 characters after an R1 hit and before the next full stop — e.g. *"ABCD" and "EFGH"*

Notes
- Weak label: a suspension is not proof of a pump; many orders cite delinquent filings or doubts about public information. The episode window is a hypothesis window.
- Survivorship: Yahoo no longer serves most suspended OTC issuers, so the loaded sample is biased towards issuers that kept trading.
- The list / RSS parser was written without access to the live pages (sec.gov answered 403 to an undeclared client on 2026-10-03) and is tested on synthetic pages only; check it against the first real download.
- Choices not fixed by the spec: bars are the trading days present in the price file (irregular spacing, `meta["irregular_times"]`); rv = log(high / low); volume normaliser from pre-label bars only, at least 10 of them (`min_pre_bars`); at least 20 bars per frame (`min_bars`); one frame per ticker when an order names several; prices are read from the cache first, yfinance is called only for tickers with neither a cache file nor an entry in `prices/_unavailable.txt`.
