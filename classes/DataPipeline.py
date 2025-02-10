import ephem as ep
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level; can be overridden via command-line arguments
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)

class DataPipeline:
    def __init__(self, company_name=None, manual_ipo_date=None):
        self.dataset = {}
        self.company_name = company_name
        self.manual_ipo_date = manual_ipo_date
        self.planets = ['Sun', 'Mercury', 'Venus', 'Moon', 'Mars', 'Jupiter', 'Saturn']
        self.aspects = {
            'conjunction': {'angle': 0, 'orb': 4, 'weight': 1},
            'opposition': {'angle': 180, 'orb': 4, 'weight': 1},
            'trine': {'angle': 120, 'orb': 4, 'weight': 1},
            'square': {'angle': 90, 'orb': 4, 'weight': 1}
        }

    def get_ipo_date(self, ticker: str) -> dt.date:
        if self.manual_ipo_date is not None:
            logging.info(f"Using manual IPO date for {ticker}: {self.manual_ipo_date}")
            return self.manual_ipo_date

        try:
            stock = yf.Ticker(ticker)
            ipo_date = stock.info.get('ipoDate')
            if ipo_date:
                ipo_date_parsed = dt.datetime.strptime(ipo_date, '%Y-%m-%d').date()
                logging.info(f"Retrieved IPO date for {ticker} from yfinance: {ipo_date_parsed}")
                return ipo_date_parsed
            logging.info(f"IPO date not found in info for {ticker}. Inferring from historical data...")
            historical_data = stock.history(period="max")
            if not historical_data.empty:
                inferred_ipo_date = historical_data.index[0].to_pydatetime().date()
                logging.info(f"Inferred IPO date for {ticker}: {inferred_ipo_date}")
                return inferred_ipo_date
            else:
                raise ValueError(f"No historical data available for ticker {ticker}.")
        except Exception as e:
            logging.error(f"Error retrieving IPO date for {ticker}: {e}")
            return None

    def normalize_angle(self, angle: float) -> float:
        return angle % 360

    def angular_separation(self, angle1: float, angle2: float) -> float:
        diff = abs(self.normalize_angle(angle1) - self.normalize_angle(angle2))
        return min(diff, 360 - diff)

    def gaussian_weight(self, x: float, mean: float, std: float) -> float:
        if x > 3 * std:
            return 0
        return norm.pdf(x, mean, std) / norm.pdf(0, mean, std)

    def calculate_aspect_intensity(self, angle1: float, angle2: float, aspect_angle: float) -> float:
        separation = self.angular_separation(angle1, angle2)
        aspect_separation = abs(separation - aspect_angle)
        return self.gaussian_weight(aspect_separation, 0, 5)

    def get_planet_position(self, planet_name: str, date: dt.datetime) -> float:
        observer = ep.Observer()
        observer.date = date
        planet = getattr(ep, planet_name)()
        planet.compute(observer)
        ecliptic = ep.Ecliptic(planet)
        position = self.normalize_angle(np.degrees(ecliptic.lon))
        logging.debug(f"{planet_name} position on {date.date()}: {position}")
        return position

    def calculate_state_features(self, natal_positions, transit_positions) -> pd.Series:
        features = {}
        for natal_planet in self.planets:
            for transit_planet in self.planets:
                if natal_planet == transit_planet:
                    continue
                for aspect_name, aspect_info in self.aspects.items():
                    intensity = self.calculate_aspect_intensity(
                        natal_positions[natal_planet],
                        transit_positions[transit_planet],
                        aspect_info['angle']
                    )
                    weighted_intensity = intensity * aspect_info['weight']
                    if weighted_intensity > 0.1:
                        feature_name = f"{natal_planet}_{transit_planet}_{aspect_name}"
                        features[feature_name] = weighted_intensity
        return pd.Series(features)

    def generate_training_data(self, start_date: dt.datetime, end_date: dt.datetime, ipo_date: dt.datetime) -> pd.DataFrame:
        logging.info("Calculating natal positions...")
        natal_positions = {planet: self.get_planet_position(planet, ipo_date) for planet in self.planets}
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        logging.info("Generating features...")
        data = []
        for date in tqdm(date_range, desc="Generating Training Data"):
            transit_positions = {planet: self.get_planet_position(planet, date) for planet in self.planets}
            features = self.calculate_state_features(natal_positions, transit_positions)
            features.name = date
            data.append(features)
        df = pd.DataFrame(data)
        df.index = date_range
        df = df.fillna(0)
        return df
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD using percentage changes instead of raw prices.
    
        Args:
        data (pd.DataFrame): DataFrame with OHLC percentage changes
        fast_period (int): Period for fast EMA
        slow_period (int): Period for slow EMA
        signal_period (int): Period for signal line
        """
        try:
            # Use Close percentage changes for MACD calculation
            price_series = data['Close']
        
            # Calculate EMAs on the percentage changes
            fast_ema = price_series.ewm(span=fast_period, adjust=False).mean()
            slow_ema = price_series.ewm(span=slow_period, adjust=False).mean()
        
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
        
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
            # Calculate MACD histogram
            macd_hist = macd_line - signal_line
        
            # Create DataFrame with MACD values
            macd_data = pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Hist': macd_hist
            }, index=data.index)
        
            # Drop the initial periods where MACD couldn't be properly calculated
            macd_data = macd_data.iloc[slow_period-1:]
        
            logging.debug(f"MACD calculated on percentage changes. Shape: {macd_data.shape}")
        
            return macd_data
        
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return pd.DataFrame(index=data.index)

    def add_technical_indicators(self, data):
    
        try:
            # Make a copy of the input data
            df = data.copy()
        
            # Calculate MACD
            macd_data = self.calculate_macd(df)
        
            # Inner join to keep only rows where we have both price data and valid MACD
            df = df.join(macd_data, how='inner')
        
            return df
        
        except Exception as e:
            logging.error(f"Error adding technical indicators: {e}")
            return data
        
    def calculate_price_changes(self, df):
        """
        Calculate percentage changes for OHLC data and clip to [-1, 1] range
        """
        # Create a copy to avoid modifying the original
        price_df = df.copy()
    
        # Calculate percentage changes for each price column
        for col in ['Open', 'High', 'Low', 'Close']:
            # Calculate percentage change relative to previous day's open
            prev_open = price_df['Open'].shift(1)
            price_df[f'{col}_pct'] = ((price_df[col] - prev_open) / prev_open.abs()).clip(-1, 1) * 100
        
        # Drop the first row which will have NaN values
        price_df = price_df.dropna()
    
        # Drop original OHLC columns
        #price_df = price_df.drop(['Open', 'High', 'Low', 'Close'], axis=1)

        price_df = price_df.rename(columns={
        'Open': 'Open-orig',
        'High': 'High-orig',
        'Low': 'Low-orig',
        'Close': 'Close-orig'
        })
        # Rename percentage columns to original names for compatibility
        price_df = price_df.rename(columns={
        'Open_pct': 'Open',
        'High_pct': 'High',
        'Low_pct': 'Low',
        'Close_pct': 'Close'
        })
    
        return price_df
    
    def generate_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            logging.error(f"No data fetched for {symbol}.")
            raise ValueError(f"No data fetched for {symbol}.")

        # If there's a MultiIndex (e.g., (Date, Ticker)), drop the second level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Reset the index to ensure a single-level index
        df = df.reset_index()

        # Ensure the date column matches the index format of astro_df
        df.set_index("Date", inplace=True)

        # Keep only the OHLC columns
        df = df[['Open', 'High', 'Low', 'Close']]
        
        df = self.calculate_price_changes(df)
        
        df = self.add_technical_indicators(df)
        
        return df

    def combine_datasets(self, astro_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        combined = pd.merge(astro_df, price_df, left_index=True, right_index=True, how='inner')
        return combined

    def min_max_scale_pa(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler(feature_range=(0.001, 1))
        columns_to_scale = ['High', 'Low', 'Open', 'Close']
        
        # Check if the columns exist to avoid errors
        existing_columns = [col for col in columns_to_scale if col in combined_df.columns]
        if not existing_columns:
            logging.warning("No price columns to scale.")
            return combined_df

        scaled_values = scaler.fit_transform(combined_df[existing_columns])
        combined_df[existing_columns] = scaled_values

        return combined_df

    def populate_dataset(self, ticker_date_dict: dict):
        """
        Populates self.dataset using a dictionary where the key is the ticker symbol
        and the value is the IPO date (can be None to infer).

        Parameters:
        - ticker_date_dict (dict): Dictionary with ticker symbols as keys and IPO dates as values.
        """
        self.dataset = {}
        logging.info("Starting dataset population...")
        for ticker, manual_date in tqdm(ticker_date_dict.items(), desc="Processing Tickers"):
            logging.info(f"\nProcessing ticker: {ticker}")
            # Determine IPO date
            if manual_date:
                ipo_date = manual_date
                logging.info(f"Using manual IPO date for {ticker}: {ipo_date}")
            else:
                ipo_date = self.get_ipo_date(ticker)
                if ipo_date is None:
                    logging.warning(f"Skipping {ticker} due to missing IPO date.")
                    continue

            # Define date range for data generation
            start_date = dt.datetime.combine(ipo_date, dt.datetime.min.time())
            end_date = dt.datetime.now()

            try:
                # Generate astrological features
                astro_df = self.generate_training_data(start_date, end_date, start_date)

                # Generate price data
                price_df = self.generate_price_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                # Combine astrological features with price data
                combined_df = self.combine_datasets(astro_df, price_df)

                # Scale the price columns
                scaled_df = self.min_max_scale_pa(combined_df)

                # Store the processed DataFrame in the dataset
                self.dataset[ticker] = scaled_df
                logging.info(f"Successfully processed and added data for {ticker}.")

            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")

        logging.info("Dataset population completed.")


def main():
    # Optional: Add argument parsing for verbosity
    parser = argparse.ArgumentParser(description="Data Pipeline for Tickers")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--loglevel', type=str, default='INFO', help='Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # Configure logging based on arguments
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    logging.getLogger().setLevel(numeric_level)
    
    # Define the tickers and their manual IPO dates
    tickers_info = {
        # Tech
        'AAPL': dt.date(1980, 12, 12),
        'MSFT': dt.date(1986, 3, 13),
        #'AMZN': dt.date(1997, 5, 15),

        # Automobile
        #'TSLA': dt.date(2010, 6, 29),
        'TM': dt.date(1980, 12, 12),
        #'RACE': dt.date(2015, 10, 21),
        #'GM': dt.date(2010, 11, 17),
        'F': dt.date(1956, 1, 18),
        
        # Pharma
        #'PFE': dt.date(1980, 12, 12),
        'LLY': dt.date(1970, 7, 9),
        'JNJ': dt.date(1944, 9, 25),
        #'NVO': dt.date(1974, 5, 17),
        #'MRK': dt.date(1941, 1, 1),
        
        #Industrials
        #'HON': dt.date(1985, 9, 19),
        'CAT': dt.date(1929, 12, 2),
        #'GE': dt.date(1892, 6, 23),
        #'UNP': dt.date(1969, 6, 20),
        'DE': dt.date(1933, 6, 29),
        
        #Insurance
        #'ALL' : dt.date(1993, 6, 3),
        'PGR' : dt.date(1987, 1, 6),
        #'TRV' : dt.date(1980, 3, 17),
        'ACGL' : dt.date(1995, 9, 13),
        #'UNH' : dt.date(1984, 10, 17),
        
        #Integrated Oil & Gas
        'XOM' : dt.date(1980, 3, 17),
        'CVX' : dt.date(1921, 6, 24),
        #'SHEL' : dt.date(2005, 7, 20),
        #'BP' : dt.date(1954, 3, 29),

        #Aerospace
        'BA' : dt.date(1934, 9, 5),
        #'RTX' : dt.date(1934, 9, 4),
        'LMT' : dt.date(1961, 10, 11),
        #'GD' : dt.date(1952, 4, 25),
        'AXON' : dt.date(2001, 5, 8),
        
        #Entertainment
        'DIS' : dt.date(1957, 11, 12),
        #'NFLX' : dt.date(2002, 5, 23),
        #'WBD' : dt.date(2005, 7, 6),
        'SPOT' : dt.date(2018, 4, 3),
        
        
        #Cable and Satalite
        'CMCSA' : dt.date(1972, 6, 29),
        'T' : dt.date(1983, 11, 21),
        'SIRI' : dt.date(1968, 1, 16),
        #'CHTR' : dt.date(2009, 12, 2),
        
        #Finance
        'JPM' : dt.date(1969, 3, 5),
        'V' : dt.date(2008, 3, 25),
        'MA' : dt.date(2006, 5, 25),
        'FIS' : dt.date(2001, 6, 20),
        #'TOAST' : dt.date(2021, 9, 22),
        
        #Real Estate
        #'PLD' : dt.date(1997, 11, 21),
        'EQIX' : dt.date(2000, 8, 11),
        #'SPG' : dt.date(1993, 12, 14),
        'WELL' : dt.date(1998, 5, 13),
        #'BX' : dt.date(2007, 6, 22),
        'BLK' : dt.date(1999, 10, 1),
        
        #Retail:
        'WMT' : dt.date(1972, 8, 25),
        'COST' : dt.date(1985, 12, 5),
        #'DG' : dt.date(2009, 11, 13),
        'TGT' : dt.date(1969, 9, 8),
        
        #Grocery
        'KR' : dt.date(1928, 1, 26),
        #'SFM' : dt.date(2013, 8, 1),
        #'ACI' : dt.date(2020, 6, 26),
        
        #Food
        'MCD' : dt.date(1965, 4, 21),
        'SBUX' : dt.date(1992, 6, 26),
        'DPZ' : dt.date(2004, 7, 13),
        #'YUM' : dt.date(1997, 9, 17),
        #'DRI' : dt.date(1995, 5, 9),
        #'TXRH' : dt.date(2004, 10, 5),
        
        #Household 
        'PG' : dt.date(1950, 3, 22),
        #'CL' : dt.date(1930, 3, 31),
        #'CLX' : dt.date(1928, 8, 1),
        'CHD' : dt.date(1980, 3, 17),
        #'WDFC' : dt.date(1973, 1, 17)
    }

    # Instantiate the DataPipeline class
    pipeline = DataPipeline(company_name="MultiCompany")
    ticker_to_int = {ticker: i for i, ticker in enumerate(tickers_info.keys())}
    # Populate the dataset with the tickers_info
    pipeline.populate_dataset(tickers_info)

    # Check if any data was processed
    if not pipeline.dataset:
        logging.warning("No data was processed. Exiting.")
        return

    # Consolidate all individual DataFrames into a single DataFrame
    logging.info("Consolidating data from all tickers...")
    all_data = []
    for ticker, df in pipeline.dataset.items():
        df_copy = df.copy()
        df_copy['Ticker'] = ticker_to_int.get(ticker)  # Add a Ticker column
        all_data.append(df_copy)

    # Concatenate all DataFrames at once to avoid fragmentation
    combined_df = pd.concat(all_data)

    # Ensure the index is named 'Date' before resetting
    combined_df.index.name = 'Date'
    combined_df.reset_index(inplace=True)

    # Define the desired order for specific columns
    desired_first = ['Date']
    desired_last = ['High', 'Low', 'Open', 'Close', 'High-orig', 'Low-orig', 'Open-orig', 'Close-orig', 'Ticker']

    # Get all current columns
    all_columns = combined_df.columns.tolist()

    # Extract columns that are neither in desired_first nor desired_last
    middle_columns = [col for col in all_columns if col not in desired_first + desired_last]

    # Define the new column order
    new_column_order = desired_first + middle_columns + desired_last

    # Reorder the DataFrame columns
    combined_df = combined_df[new_column_order]

    # Save the combined DataFrame to a CSV file
    output_csv = 'full_df.csv'
    logging.info(f"Saving the consolidated data to {output_csv}...")
    combined_df.to_csv(output_csv, index=False)
    logging.info("Data successfully saved to CSV.")

if __name__ == "__main__":
    main()