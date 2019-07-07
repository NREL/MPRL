# ========================================================================
#
# Imports
#
# ========================================================================
import pandas as pd
from scipy import interpolate as interp


# ========================================================================
#
# Functions
#
# ========================================================================
def interpolate_df(x, name, fp):
    """Interpolate a dataframe

    :param x: the x-coordinates at which to evaluate the interpolated values
    :type x: array
    :param name: the name of the column to use in the dataframe for the x-coordinate
    :type name: str
    :param fp: the dataframe containing the y-coordinates
    :type fp: DataFrame
    :returns: the interpolated dataframe
    :rtype: DataFrame
    """
    df = pd.DataFrame({name: x})
    for col in fp.columns:
        f = interp.interp1d(fp[name], fp[col], kind="linear", fill_value="extrapolate")
        df[col] = f(df[name])

    return df


# ========================================================================
def evaluate_agent(env, agent):
    """Evaluate an agent in an engine environment.
    
    :param env: engine environment
    :type env: Environment
    :param agent: agent
    :type agent: Agent
    :returns: dataframe of history, total rewards
    """

    engine = env.envs[0]

    # Save all the history
    df = pd.DataFrame(
        0.0,
        index=engine.history.index,
        columns=list(
            dict.fromkeys(
                list(engine.history.columns)
                + engine.observables
                + engine.internals
                + engine.actions
                + engine.histories
                + ["rewards"]
            )
        ),
    )
    df[engine.histories] = engine.history[engine.histories]
    df.loc[0, ["rewards"]] = [engine.p0 * engine.history.dV.loc[0]]

    # Evaluate actions from the agent in the environment
    obs = env.reset()
    df.loc[0, engine.observables] = obs
    df.loc[0, engine.internals] = engine.current_state[engine.internals]
    for index in engine.history.index[1:]:
        action, _ = agent.predict(obs)
        obs, reward, done, info = env.step(action)

        # save history
        df.loc[index, engine.actions] = action
        df.loc[index, engine.internals] = info[0]["internals"]
        df.loc[index, ["rewards"]] = reward
        if done:
            break
        df.loc[index, engine.observables] = obs

    df = df.loc[:index, :]

    return df, df.rewards.sum()
