import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Level2:
    def __init__(self, organism, poppulation) -> None:
        self.organism = organism
        self.poppulation = poppulation
        # pygame.init()
        self.width = 1200
        self.height = 800
        # self.game_display = pygame.display.set_mode((self.width, self.height))
        # pygame.display.set_caption("Level 2")
        # self.clock = pygame.time.Clock()
        self.food_color = (255, 255, 255)
        self.food_size = 100
        self.player_size = 10
        self.max_age = 200
        self.speed = 5
        self.food_num = 5
        self.food_age = 50
        self.food = {
            0: {"position": (self.width // 3, self.height // 3)},
            1: {"position": (2 * self.width // 3, self.height // 3)},
            2: {"position": (self.width // 3, 2 * self.height // 3)},
            3: {"position": (2 * self.width // 3, 2 * self.height // 3)},
            4: {"position": (self.width // 2, self.height // 2)},
        }

        self.players = {}
        self.player_radius = 300
        self.game_over = False
        self.edge_guards = []
        self._place_edge_wall()
        self._spawn_players()
        while True:
            self._play_game()

    def _place_edge_wall(self):
        x_num = self.width // self.player_radius
        y_num = self.height // self.player_radius
        for i in range(x_num * 2 + 1):
            self.edge_guards.append((i * self.player_radius / 2, 0))
            self.edge_guards.append((i * self.player_radius / 2, self.height))
        for i in range(y_num * 2 + 1):
            self.edge_guards.append((0, i * self.player_radius / 2))
            self.edge_guards.append((self.width, i * self.player_radius / 2))

    def _spawn_edge_wall(self):
        for guard in self.edge_guards:
            pygame.draw.circle(
                self.game_display,
                self.food_color,
                guard,
                5,
            )

    def _place_food(self):
        for i in range(self.food_num):
            pygame.draw.circle(
                self.game_display,
                self.food_color,
                self.food[i]["position"],
                self.food_size // 2,
                1,
            )

    def _spawn_food(self, food_index):
        self.food[food_index]["position"] = (
            random.randint(10, self.width - 10),
            random.randint(10, self.height - 10),
        )
        self.food[food_index]["age"] = 0

    def _place_player(self):
        x = random.randint(0, self.width - self.player_size)
        y = random.randint(0, self.height - self.player_size)
        return (x, y)

    def _spawn_players(self):
        for i in range(self.poppulation):
            player = self.organism().load(i)
            position = self._place_player()
            self.players[i] = {
                "player": player,
                "position": position,
                "age": 0,
                "color": random.randint(0, 255),
                "state_1": np.zeros(200),
                "state_2": np.zeros(200),
                "state_3": np.zeros(200),
                "action": torch.tensor(
                    np.zeros(203), dtype=torch.float32, device="cuda"
                ),
            }
        self._update_ui()

    def _spawn_new_player(self, index):
        random_index = random.randint(0, self.poppulation - 1)
        player = self.organism().load(random_index)
        position = self._place_player()
        self.players[index] = {
            "player": player,
            "position": position,
            "age": 0,
            "color": random.randint(0, 255),
            "state_1": np.zeros(200),
            "state_2": np.zeros(200),
            "state_3": np.zeros(200),
            "action": torch.tensor(np.zeros(203), dtype=torch.float32, device="cuda"),
        }

    def _get_distance(self, player_position, object_position):
        dist_x = object_position[0] - player_position[0]
        dist_y = object_position[1] - player_position[1]
        net_dist = ((dist_x**2) + (dist_y**2)) ** 0.5
        if net_dist > self.player_radius:
            return None, None
        return dist_x, dist_y

    def _update_states(self):
        for i in range(self.poppulation):
            self._update_player_state(player_index=i)

    def _update_player_state(self, player_index):
        player_position = self.players[player_index]["position"]
        current_state = np.zeros(200)
        total_objects = 0
        for i in range(self.food_num):
            if total_objects > 49:
                break
            food_distance_x, food_distance_y = self._get_distance(
                player_position=player_position,
                object_position=self.food[i]["position"],
            )
            if food_distance_x:
                current_state[total_objects * 4 : total_objects * 4 + 4] = [
                    food_distance_x,
                    food_distance_y,
                    255,
                    -10,
                ]
                total_objects = total_objects + 1
        for guard in self.edge_guards:
            if total_objects > 49:
                break
            dist_x, dist_y = self._get_distance(
                player_position=player_position,
                object_position=guard,
            )
            if dist_x:
                current_state[total_objects * 4 : total_objects * 4 + 4] = [
                    dist_x,
                    dist_y,
                    0,
                    -100,
                ]
                total_objects = total_objects + 1
        for i in range(self.poppulation):
            if total_objects > 49:
                break
            if not i == player_index:
                dist_x, dist_y = self._get_distance(
                    player_position=player_position,
                    object_position=self.players[i]["position"],
                )
                if dist_x:
                    current_state[total_objects * 4 : total_objects * 4 + 4] = [
                        dist_x,
                        dist_y,
                        self.players[i]["color"],
                        i,
                    ]
                    total_objects = total_objects + 1
        self.players[player_index]["state_1"] = self.players[player_index]["state_2"]
        self.players[player_index]["state_2"] = self.players[player_index]["state_3"]
        self.players[player_index]["state_3"] = current_state

    def _get_state(self, player_index):
        return np.concatenate(
            (
                self.players[player_index]["state_1"],
                self.players[player_index]["state_2"],
                self.players[player_index]["state_3"],
            )
        )

    def _check_for_collision(self, player_index):
        if (
            self.players[player_index]["position"][0] <= 0
            or self.players[player_index]["position"][0] >= self.width
            or self.players[player_index]["position"][1] <= 0
            or self.players[player_index]["position"][1] >= self.height
        ):
            self._spawn_new_player(player_index)
        else:
            self._check_for_age(player_index)

    def _check_for_age(self, player_index):
        if self.players[player_index]["age"] >= self.max_age:
            self._spawn_new_player(player_index)
        else:
            self._check_for_food(player_index)

    def _check_for_food(self, player_index):
        position = self.players[player_index]["position"]
        for i in range(self.food_num):
            confirm1 = position[0] > (self.food[i]["position"][0] - self.food_size / 2)
            confirm2 = position[0] < (self.food[i]["position"][0] + self.food_size / 2)
            confirm3 = position[1] > (self.food[i]["position"][1] - self.food_size / 2)
            confirm4 = position[1] < (self.food[i]["position"][1] + self.food_size / 2)
            if confirm1 and confirm2 and confirm3 and confirm4:
                # print("player ", player_index, " ate food ", i)
                player = self.players[player_index]["player"]
                player.save(index=player_index)
                new_player = player.reproduce(player_index)
                position = self._place_player()
                self.players[player_index] = {
                    "player": new_player,
                    "position": position,
                    "age": 0,
                    "color": random.randint(0, 255),
                    "state_1": np.zeros(200),
                    "state_2": np.zeros(200),
                    "state_3": np.zeros(200),
                    "action": torch.tensor(
                        np.zeros(203), dtype=torch.float32, device="cuda"
                    ),
                }
                break

    def _train_players(self):
        losses = []
        for i in range(self.poppulation):
            player = self.players[i]["player"]
            output_tensor = self.players[i]["action"]
            target_tensor = torch.cat(
                (
                    output_tensor[:3],
                    torch.tensor(
                        self.players[i]["state_3"], device="cuda", dtype=torch.float32
                    ),
                )
            )
            output_tensor.requires_grad = True
            criterion = nn.MSELoss()
            loss = criterion(output_tensor, target_tensor)
            if torch.isinf(loss):
                # print("infinite loss for player ", i)
                # print(output_tensor)
                # print(target_tensor)
                self._spawn_new_player(i)
                indices = torch.where((target_tensor > 1200) | (target_tensor < -1200))
                print(indices)
            else:
                # if i == 0:
                #     print(loss.item())
                optimizer = optim.Adam(player.parameters(), lr=0.01)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        print(sum(losses) / len(losses))

    def _play_game(self):
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         self._save_players()
        #         pygame.quit()
        #         quit()
        for i in range(self.poppulation):
            player = self.players[i]["player"]
            output = player.get_action(x=self._get_state(player_index=i))
            self.players[i]["action"] = output.clone()
            # output[0] = torch.tanh(output[0])
            # output[1] = torch.tanh(output[1])
            # output[2] = torch.sigmoid(output[2]) * 255
            output = output.cpu().numpy().tolist()

            self.players[i]["position"] = (
                self.players[i]["position"][0] + output[0] * self.speed,
                self.players[i]["position"][1] + output[1] * self.speed,
            )
            self.players[i]["color"] = output[2]
            self._check_for_collision(i)
        self._update_states()
        self._train_players()
        self._update_ui()
        # self.clock.tick(30)  # Frame rate

    def _update_ui(self):
        return
        self.game_display.fill((0, 0, 0))
        self._place_food()
        self._spawn_edge_wall()
        for i in range(self.poppulation):
            player_pos = self.players[i]["position"]
            if i == 0:
                pygame.draw.circle(
                    self.game_display,
                    (255, 255, 0),
                    player_pos,
                    self.player_size // 2,
                )
                pygame.draw.circle(
                    self.game_display,
                    (255, 255, 150),
                    player_pos,
                    self.player_radius,
                    1,
                )
            else:
                pygame.draw.circle(
                    self.game_display,
                    (150, self.players[i]["color"], 150),
                    player_pos,
                    self.player_size // 2,
                )

            self.players[i]["age"] += 1
        pygame.display.update()

    def _save_players(self):
        pass
        # for i in range(self.poppulation):
        #     player = self.players[i]["player"]
        #     player.save(i)
