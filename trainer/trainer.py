import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, tokenizer, optimizer, criterion, device, entity_markers,
                 labels=['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']):
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.tokenizer = tokenizer
        self.E1_s = entity_markers[0]
        self.E2_s = entity_markers[1]
        self.labels = labels
        self.label_2_id = {l: i for i, l in enumerate(labels)}
        self.id_2_label = {i: l for i, l in enumerate(labels)}

    def forward_step(self, input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks, labels,
                     marks_only=True,
                     entity_and_marks=False, is_eval=False):

        out = self.model(input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks, marks_only,
                         entity_and_marks)

        loss = self.criterion(out, labels)
        if not is_eval:
            loss.backward()
            self.optimizer.step()
        return loss, out

    def prepare_batch(self, batch):
        tokenized_data = self.tokenizer(batch['text'], padding=True)

        input_ids = torch.tensor(tokenized_data['input_ids'], device=self.device)
        attention_masks = torch.tensor(tokenized_data['attention_mask'], device=self.device)

        em1_s = torch.tensor([(ids == self.E1_s).nonzero().item() for ids in input_ids], device=self.device)
        entity_1 = em1_s + 1

        em2_s = torch.tensor([(ids == self.E2_s).nonzero().item() for ids in input_ids], device=self.device)
        entity_2 = em2_s + 1

        label = torch.tensor([self.label_2_id[label] for label in batch['label']], device=self.device)

        return input_ids, entity_1, entity_2, em1_s, em2_s, attention_masks, label

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
                input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks, label = self.prepare_batch(
                    batch)
                loss, _ = self.forward_step(input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks,
                                            label)
                running_loss += loss.item()

                tqdm_train_loader.update()
            epoch_progress.update()
            tqdm_train_loader.reset()
