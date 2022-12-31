from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model

    def forward_step(self, input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks, labels, is_eval):

        out = self.model(input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks, labels, is_eval)

        loss = self.criterion(out, labels)
        if not is_eval:
            loss.backward()
            self.optimizer.step()
        return loss, out

    def train(self, train_loader, test_loader, num_epochs, eval_step, best_valid_loss=float('inf')):
        # initialize running values
        running_loss = 0.0
        valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        # training loop
        self.model.train()
        self.model.to(self.device)
        epoch_progress = tqdm(range(num_epochs))
        tqdm_train_loader = tqdm(train_loader)
        for epoch in range(num_epochs):

            for batch in train_loader:
                text, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks = batch
                loss, _ = self.forward_step(text, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks)
                running_loss += loss.item()

                tqdm_train_loader.update()
            epoch_progress.update()
            tqdm_train_loader.reset()
