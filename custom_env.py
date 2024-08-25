# Jax: https://jax.readthedocs.io/en/latest/index.html
# Chex: https://chex.readthedocs.io/en/latest/index.html
# Flex: https://flax.readthedocs.io/en/latest/index.html
# JaxMARL: https://github.com/FLAIROx/JaxMARL
# PettingZoo: https://pettingzoo.farama.org/tutorials/custom_environment/

import time

import jax
import jax.numpy as jnp

import numpy as np

from jaxmarl.environments.multi_agent_env import MultiAgentEnv

import chex

from gymnax.environments.spaces import Box, Discrete

from flax import struct

from typing import Tuple, Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib

@struct.dataclass
class State:
    """ Basic state """
    escape_pos: jax.Array
    guard_pos: jax.Array
    prisoner_pos: jax.Array
    done: jax.Array
    timestep: int
    
class CustomEnv(MultiAgentEnv):

    metadata = {
        "name": "custom_env_v0"
    }

    def __init__(self, max_steps=25):
        self.agents = ("prisoner", "guard")
        self.actions = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}
        self.max_steps = max_steps

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, jax.Array]:
        """Return dictionary of agent observations"""

        @partial(jax.vmap)
        def _op_obs(vec: jax.Array) -> jax.Array:
            return vec[0] + 7 * vec[1]
        
        state_stacked = jnp.stack([
            state.escape_pos,
            state.guard_pos,
            state.prisoner_pos
        ]) 

        obs = _op_obs(state_stacked)
        return {a: obs for a in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: jax.Array) -> Tuple[Dict, State]:
        key_x, key_y = jax.random.split(key)

        escape_pos = jnp.concatenate([
            jax.random.randint(key_x, shape=(1,), minval=2, maxval=5),
            jax.random.randint(key_y, shape=(1,), minval=2, maxval=5)
        ])

        state = State(
            escape_pos=escape_pos,
            guard_pos=jnp.array((6, 6)), 
            prisoner_pos=jnp.array((0, 0)), 
            done=jnp.full(len(self.agents), False),
            timestep=0
        )
        
        return self.get_obs(state), state

    def _get_new_pos(self, state: State, actions: Dict) -> Tuple[jax.Array, jax.Array]:
        acts = [actions["prisoner"], actions["guard"]]
        prisoner_new_pos = [state.prisoner_pos[0].item(), state.prisoner_pos[1].item()]
        guard_new_pos = [state.guard_pos[0].item(), state.guard_pos[1].item()]
        new_pos = (prisoner_new_pos, guard_new_pos)

        for i in range(len(self.agents)):
            if acts[i] == self.actions["LEFT"] and new_pos[i][0] > 0:
                new_pos[i][0] -= 1
            elif acts[i] == self.actions["RIGHT"] and new_pos[i][0] < 6:
                new_pos[i][0] += 1
            elif acts[i] == self.actions["UP"] and new_pos[i][1] > 0:
                new_pos[i][1] -= 1
            elif acts[i] == self.actions["DOWN"] and new_pos[i][1] < 6:
                new_pos[i][1] += 1
        
        prisoner_new_pos, guard_new_pos = new_pos
        return jnp.array(prisoner_new_pos), jnp.array(guard_new_pos)

    def _get_reward(self, prisoner_pos, guard_pos, escape_pos, timestep) -> Tuple[Dict, jax.Array]:
        rewards = {a: 0 for a in self.agents}
        dones = None

        # Check termination condition
        if jnp.all(guard_pos == prisoner_pos):
            rewards = {"guard": 1, "prisoner": -1}
            dones = jnp.full(len(self.agents), True)
        elif jnp.all(escape_pos == prisoner_pos):
            rewards = {"guard": -1, "prisoner": 1}
            dones = jnp.full(len(self.agents), True)
        
        # Check truncation condition
        if timestep >= self.max_steps:
            rewards = {"guard": 0, "prisoner": 0}
            dones = jnp.full(len(self.agents), True)
        
        return rewards, dones

    #@partial(jax.jit, static_argnums=[0])
    def step(self, key: jax.Array, state: State, actions: Dict) -> Tuple[Dict, State, Dict, jax.Array, Dict]:
        prisoner_new_pos, guard_new_pos = self._get_new_pos(state, actions)

        rewards, dones = self._get_reward(prisoner_new_pos, guard_new_pos, state.escape_pos, state.timestep)
        if dones == None:
            dones = jnp.full(len(self.agents), state.timestep >= self.max_steps) 
        
            state = state.replace(
                guard_pos=guard_new_pos,
                prisoner_pos=prisoner_new_pos,
                done=dones,
                timestep=state.timestep + 1
            )
            
            obs = self.get_obs(state)
        else:
            obs, state = self.reset(key)

        info = {}
        
        return obs, state, rewards, dones, info

    def render(self, state: State) -> None:
        grid = np.full((7, 7), " ")
        grid[state.escape_pos[1].item(), state.escape_pos[0].item()] = "E"
        grid[state.guard_pos[1].item(), state.guard_pos[0].item()] = "G"
        grid[state.prisoner_pos[1].item(), state.prisoner_pos[0].item()] = "P"
        print(f"{grid} \n")
        
        time.sleep(0.5)


if __name__ == "__main__":
    key = jax.random.key(0)
    env = CustomEnv()
    obs, state = env.reset(key)
    env.render(state)

    for i in range(env.max_steps):
        key, key_s, key_a = jax.random.split(key, 3)
        key_a0, key_a1 = jax.random.split(key_a)

        a0 = jax.random.randint(key_a0, shape=(1,), minval=0, maxval=4)
        a1 = jax.random.randint(key_a1, shape=(1,), minval=0, maxval=4)
        actions = {env.agents[0]: a0.item(), env.agents[1]: a1.item()}

        obs, state, rewards, dones, info = env.step(key_s, state, actions)

        env.render(state)
