import pandas as pd
import os
import numpy as np
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
import itertools
from datetime import timedelta 
from datetime import datetime
from finrl.config_tickers import CRYPTO_TICKERS
from finrl.config import INDICATORS

TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2022-12-29'
TRADE_START_DATE = '2022-12-29'
TRADE_END_DATE = '2022-12-31'

TIMESTEPS = 10000

PORTFOLIO = CRYPTO_TICKERS

def load_data():
    L=400
    # download more date to compute tech indicator like macd
    date = datetime.strptime(TRAIN_START_DATE, "%Y-%m-%d")
    date = date - timedelta(days=L)
    datetime.strftime(date, "%Y-%m-%d")
    df = YahooDownloader(start_date = date,
                        end_date = TRADE_END_DATE,
                        ticker_list = PORTFOLIO).fetch_data()
    return df


def process(df, phase='test', state=None):
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=False,
                    use_turbulence=False,
                    user_defined_feature = False)

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)

    train.to_csv('train.csv')
    trade.to_csv('trade.csv')

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    env_train, _ = StockTradingEnv(df = train, **env_kwargs).get_sb_env()

    if_using_a2c = True
    if_using_ddpg = True
    if_using_ppo = True
    if_using_td3 = True
    if_using_sac = True

    if phase=='train':
        agent = DRLAgent(env = env_train)
        model_a2c = agent.get_model("a2c")

        trained_a2c = agent.train_model(model=model_a2c, 
                                    total_timesteps=TIMESTEPS) if if_using_a2c else None

        trained_a2c.save('model.zip')

    model=model_a2c.load('model.zip')

    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)

    test_env, test_obs = e_trade_gym.get_sb_env_new(state)

    # test_env.state = (
    #                     [self.initial_amount]
    #                     + self.data.close.values.tolist()
    #                     + self.num_stock_shares
    #                     + sum(
    #                         (
    #                             self.data[tech].values.tolist()
    #                             for tech in self.tech_indicator_list
    #                         ),
    #                         [],
    #                     )
    #                 ) 

    action, _states = model.predict(test_obs, deterministic=True)
                # test_obs, rewards, dones, info = test_env.step(action)
    print(np.floor(action*env_kwargs['hmax']))
    dict()

if __name__ == '__main__':
    df = load_data()
    df.to_csv('df.csv')
    pd.read_csv('df.csv')
    state=np.concatenate([[500000,12,45,15,10,10,10],np.ones(24)])
    process(df,'train',state)