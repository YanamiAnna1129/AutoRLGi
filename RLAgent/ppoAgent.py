import numpy as np
import torch
import torch.nn.functional as F
from lpsim.agents import RandomAgent
from torch import nn
from typing import Tuple, List
import logging
from lpsim import Match, Deck
from lpsim.agents.interaction_agent import InteractionAgent_V2_0
from lpsim.server.interaction import (
    Responses,
    SwitchCardRequest, SwitchCardResponse,
    ChooseCharacterRequest, ChooseCharacterResponse,
    RerollDiceRequest, RerollDiceResponse,
    DeclareRoundEndResponse,
    ElementalTuningRequest, ElementalTuningResponse,
    SwitchCharacterRequest, SwitchCharacterResponse,
    UseSkillRequest, UseSkillResponse,
    UseCardRequest, UseCardResponse,
)
import random
from lpsim.server.elemental_reaction import check_elemental_reaction
from lpsim.server.consts import ElementType
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
deck_string = '''
default_version:4.1
character:Fischl
character:Mona
character:Collei
Gambler's Earrings*2
Wine-Stained Tricorne*2
Vanarana
Timmie*2
Rana*2
Covenant of Rock*2
The Bestest Travel Companion!*2
Changing Shifts*2
Toss-Up*2
Strategize*2
I Haven't Lost Yet!*2
Calx's Arts*2
Adeptus' Temptation*2
Lotus Flower Crisp*2
Mondstadt Hash Brown*2
Tandoori Roast Chicken
'''


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.nn.functional.leaky_relu(self.fc1(state))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):

        action_dict = {
            "end": 0,
            "sw_char 0 OMNI": 1,
            "sw_char 1 OMNI": 2,
            "sw_char 2 OMNI": 3,
            "skill 1 OMNI OMNI OMNI": 4,
            "skill 2 OMNI OMNI OMNI": 5}
        try:
            action_num = action_dict[action]
            # 如果缓冲区已满，则替换最早的经验
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action_num, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
        except KeyError:
            pass

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return torch.tensor(states, dtype=torch.float).to(device), \
            torch.tensor(actions, dtype=torch.long).to(device), \
            torch.tensor(rewards, dtype=torch.float).to(device), \
            torch.tensor(next_states, dtype=torch.float).to(device), \
            torch.tensor(dones, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.buffer)


class Agent(InteractionAgent_V2_0):
    def _get_input(self, *give_to) -> str:
        if len(self.commands) > 0:
            cmd = self.commands[0]
            self.commands = self.commands[1:]
            logging.info(f"InteractionAgent {self.player_idx}: {cmd}")
            return cmd
        if self.only_use_command:
            return "no_command"
        # print(f"InteractionAgent {self.player_idx}: ")

        return give_to[0]  # 字符串

    def _get_cmd(self, *give_to) -> Tuple[str, List[str]]:
        # print(f"getcmd获取到的give_to: {give_to}")
        while True:
            w = self._get_input(give_to[0])
            i = w.strip().split(" ")
            # print(f"i: {i}")
            if len(i) == 0:
                continue
            cmd = i[0]
            if cmd == "help":
                print(f"available commands: {list(self._cmd_to_name.keys())}")
            elif cmd == "no_command":
                return "no_command", []
            elif cmd == "print":
                self._print_requests()
            elif cmd == "verbose":
                if len(i) == 1:
                    print(f"current verbose level: {self.verbose_level}")
                else:
                    self.verbose_level = int(i[1])
                    print(f"set verbose level to {self.verbose_level}")
            elif cmd in self._cmd_to_name:
                # print(f"_getcmd返回,{self._cmd_to_name[cmd]},{i[1:]}")
                return self._cmd_to_name[cmd], i[1:]
            else:
                print("在这里解析命令不对")
                try:
                    cmd_num = int(cmd)
                    if cmd_num < 0 or cmd_num >= len(self.available_reqs):
                        print("invalid command number")
                        if self.only_use_command:
                            raise AssertionError()
                        continue
                    return self.available_reqs[cmd_num].name, i[1:]
                except ValueError:
                    print("invalid command")
                    if self.only_use_command:
                        raise AssertionError()
                    continue

    def generate_response(self, match: Match, *give_to) -> Responses | None | str:

        # if self.random_after_no_command and len(self.commands) == 0:
        #     self.random_agent.player_idx = self.player_idx
        #     return "need retry"
        self.available_reqs = [
            x for x in match.requests if x.player_idx == self.player_idx
        ]
        if len(self.available_reqs) == 0:
            print("出现了available_reqs为空的情况len(self.available_reqs) == 0")
            return None
        if self.verbose_level >= 1:
            self._print_requests()
        try:
            # print(f"141 give_to: {give_to}")
            req_name, args = self._get_cmd(give_to[0])
            # print(f"149 req_name: {req_name}, args: {args}")
            if req_name == "no_command":
                print("if req_name == no_command")
                return None
            reqs = [x for x in self.available_reqs if x.name == req_name]
            # print(f"reps: {reqs}")
            if len(reqs) == 0:
                print(f"no such request: {req_name}")
                if self.only_use_command:
                    raise AssertionError()
                return "need retry"
            if req_name == "SwitchCardRequest":
                return self.resp_switch_card(args, reqs)  # type: ignore
            if req_name == "ChooseCharacterRequest":
                return self.resp_choose_character(args, reqs)  # type: ignore
            if req_name == "RerollDiceRequest":
                return self.resp_reroll_dice(args, reqs)  # type: ignore
            if req_name == "DeclareRoundEndRequest":
                return self.resp_declare_round_end(args, reqs)  # type: ignore
            if req_name == "ElementalTuningRequest":
                return self.resp_elemental_tuning(args, reqs)  # type: ignore
            if req_name == "SwitchCharacterRequest":
                return self.resp_switch_character(args, reqs)  # type: ignore
            if req_name == "UseSkillRequest":
                return self.resp_use_skill(args, reqs)  # type: ignore
            if req_name == "UseCardRequest":
                return self.resp_use_card(args, reqs)  # type: ignore
        except Exception as e:
            print(f"17?error: {e}")
            if self.only_use_command:
                raise e
            return "need retry"


class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.eps_clip = 0.2  # PPO剪切因子
        self.K_epochs = 10  # PPO更新次数
        self.lr = 3e-4
        self.batch_size = 8
        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer()
        self.MseLoss = nn.MSELoss()
        self.agent = Agent(player_idx=0)  # 使用自定义的Agent
        self.writer = SummaryWriter()
        self.update_step = 0
    def select_action(self, state, match):

        state = torch.tensor(state, dtype=torch.float).to(device)

        reqs = list(([x for x in match.requests if x.player_idx == 0]))
        reqs.sort(key=lambda x: x.name)
        req_names = list(set([x.name for x in match.requests
                              if x.player_idx == 0]))
        req_names.sort()

        if 'SwitchCardRequest' == req_names[0]:
            return "sw_card"
        elif 'ChooseCharacterRequest' == req_names[0]:
            available_list = reqs[0].available_character_idxs
            idx = random.choice(available_list)
            return f"choose {idx}"
        elif 'RerollDiceRequest' == req_names[0]:
            return "reroll"

        elif 'DeclareRoundEndRequest' in req_names:

            action_probs = self.actor(state)  # actor 输出每个动作的概率分布
            action_probs = F.softmax(action_probs, dim=-1)
            action = torch.argmax(action_probs).item()

            if action == 0:
                return 'end'
            elif action == 1:
                return 'sw_char 0 OMNI'
            elif action == 2:
                return 'sw_char 1 OMNI'
            elif action == 3:
                return 'sw_char 2 OMNI'
            elif action == 4 or action == 5:
                return f'skill {action - 3} OMNI OMNI OMNI'

            else:
                raise NotImplementedError(f"不支持的请求类型:{req_names}")

    def update(self):
        for _ in range(self.K_epochs):
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            rewards = rewards.unsqueeze(1)  # 形状从 [64] 变成 [64, 1]
            dones = dones.unsqueeze(1)  # 形状从 [64] 变成 [64, 1]
            values = self.critic(states)
            with torch.no_grad():
                next_values = self.critic(next_states)
                td_target = rewards + self.gamma * next_values * (1 - dones)
                advantages = td_target - values  # 奖励-b价值

            # 计算旧策略下的概率
            old_action_probs = self.old_actor(states)
            old_action_probs = torch.softmax(old_action_probs, dim=-1)
            # 计算新策略下的概率
            action_probs = self.actor(states)
            action_probs = torch.softmax(action_probs, dim=-1)
            # 新旧差值
            ratio = action_probs.gather(1, actions.unsqueeze(-1)) / old_action_probs.gather(1, actions.unsqueeze(-1))
            # 剪切: 将 新旧差值 限制在 (1 +- eps_clip)
            surrogate_loss = torch.min(ratio * advantages,
                                       torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages)
            # 计算actor损失
            actor_loss = -torch.mean(surrogate_loss)
            # 计算critic损失
            critic_loss = torch.mean(self.MseLoss(values, td_target))
            # 更新actor网络
            self.actor_optimizer.zero_grad()  # actor_optimizer优化器
            actor_loss.backward()
            self.actor_optimizer.step()
            # 更新critic网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            # 更新旧策略
            self.old_actor.load_state_dict(self.actor.state_dict())
            # 记录日志
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.update_step)
            self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.update_step)
            self.update_step += 1
    def save_policy(self, filename="ppo_policy.pth"):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
            'optimizer_critic_state_dict': self.critic_optimizer.state_dict(),
        }, filename)

    def load_policy(self, filename="ppo_policy.pth"):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])


def extract_state(match, player_idx, ):
    self_hp = [match.player_tables[player_idx].characters[0].hp, match.player_tables[player_idx].characters[1].hp,
               match.player_tables[player_idx].characters[2].hp]
    enemy_hp = [match.player_tables[1 - player_idx].characters[0].hp,
                match.player_tables[1 - player_idx].characters[1].hp,
                match.player_tables[1 - player_idx].characters[2].hp]
    element_dict = {
        ElementType.NONE: 0,
        ElementType.CRYO: 1,
        ElementType.HYDRO: 2,
        ElementType.PYRO: 3,
        ElementType.ELECTRO: 4,
        ElementType.GEO: 5,
        ElementType.DENDRO: 6,
        ElementType.ANEMO: 7,
    }
    ele_list = []
    for j in range(2):
        for i in range(3):
            if len(match.player_tables[j].characters[i].element_application) == 0:
                ele_list.append(0)
            else:
                ele_list.append(element_dict[match.player_tables[j].characters[i].element_application[0]])

    return [
        match.player_tables[player_idx].active_character_idx,
        match.player_tables[1 - player_idx].active_character_idx,
        self_hp[0],
        self_hp[1],
        self_hp[2],
        enemy_hp[0],  # 5
        enemy_hp[1],
        enemy_hp[2],
        ele_list[3],  # 8
        ele_list[4],  # 9
        ele_list[5],  # 10
        ele_list[0],  # 11
        ele_list[1],  # 12
        ele_list[2],  # 13
        len(match.player_tables[player_idx].dice.colors),  # 14
        match.player_tables[player_idx].charge_satisfied,  # 15
    ]


def compute_reward(match, player_idx, action, old_state, next_state):
    reward = 0
    if action == 'end':
        if len(match.player_tables[player_idx].dice.colors) >= 3:
            print("len(match.player_tables[player_idx].dice.colors) >= 3")
            reward -= 50
    if 'skill' in action:
        print("'skill' in action")
        reward += 50
    if 'skill' in action and old_state[14] < 3:
        reward -= 50
    if 'skill' in action and action.split(' ')[1] == 2 and old_state[15]:
        reward += 100

    self_idx = old_state[0]
    enemy_idx = old_state[1]

    if next_state[8 + enemy_idx] != old_state[8 + enemy_idx]:
        reward += 50
        print("+500")
    if next_state[11 + self_idx] != old_state[11 + self_idx]:
        reward -= 50
    return reward





def main():
    EPISODES = 100000000000000000
    winner_dict = {0: 0, 1: 0, -1: 0}
    ppo_agent = PPO(state_dim=16, action_dim=5)
    agent_1 = RandomAgent(player_idx=1)
    deck0 = Deck.from_str(deck_string)
    deck1 = Deck.from_str(deck_string)

    try:
        ppo_agent.load_policy()
        print("加载已保存的 PPO 策略...")
    except FileNotFoundError:
        print("未找到已保存的策略，开始训练新的 PPO 代理...")

    for i in range(EPISODES):
        print("EPISODE: ", i)

        match = Match()
        match.set_deck([deck0, deck1])
        match.config.history_level = 10  # 不知道什么玩意
        match.start()
        match.step()
        player_idx = 0
        episode_reward = 0  # 每回合奖励

        while not match.is_game_end():
            if match.need_respond(0):
                state = extract_state(match, player_idx)
                action = ppo_agent.select_action(state, match)
                print(f"模拟的输入: {action}")
                get = ppo_agent.agent.generate_response(match, action)
                if get == "need retry":
                    for t in range(10):
                        # state = extract_state(match, player_idx)
                        if action != "sw_card" and action != "choose 0" and action != "choose 1" and action != "choose 2" and action != "reroll":
                            ppo_agent.buffer.add(state, action, reward=-20, next_state=state, done=0)
                            episode_reward += -25
                            print("重复!")
                        action = ppo_agent.select_action(state, match)
                        print(f"模拟的输入: {action}")
                        get = ppo_agent.agent.generate_response(match, action)
                        if get != "need retry":
                            break
                        if t == 9:
                            get = ppo_agent.agent.generate_response(match, "end")

                    match.respond(get)
                    match.step()


                else:
                    match.respond(get)
                    match.step()
                    if action != "sw_card" and action != "choose 0" and action != "choose 1" and action != "choose 2" and action != "reroll":
                        next_state = extract_state(match, player_idx)
                        reward = compute_reward(match, player_idx, action, state, next_state)
                        done = (1 if match.is_game_end() else 0)

                        ppo_agent.buffer.add(state, action, reward, next_state, done)

                        episode_reward += reward


            elif match.need_respond(1):
                match.respond(agent_1.generate_response(match))
                match.step()



        winner = match.winner
        winner_dict[winner] += 1
        print(f'winner is {winner_dict},reward is {episode_reward}')

        if i < 3:
            ppo_agent.batch_size = 16
        else:
            ppo_agent.batch_size = 128
        if len(ppo_agent.buffer) > ppo_agent.batch_size:
            print("==============开始更新策略...===============")
            ppo_agent.update()
        if EPISODES % 2 == 0:
            ppo_agent.save_policy()
        ppo_agent.writer.add_scalar('Reward/Episode', episode_reward, i)
        win = 1 if winner == 0 else 0
        ppo_agent.writer.add_scalar('WinRate', win, i)

if __name__ == '__main__':
    main()
