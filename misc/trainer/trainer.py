import torch
from sklearn.metrics import classification_report
from tqdm import tqdm


class Trainer:
    def __init__(self, model, tokenizer, optimizer, criterion, device, entity_markers, labels):
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

    def full_pass_step(self, batch, mode='train'):
        self.optimizer.zero_grad()
        input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks, labels = self.prepare_batch(batch)

        out = self.model(input_ids, entity_1, entity_2, entity_mark_1_s, entity_mark_2_s, masks)
        loss = self.criterion(out, labels)

        if mode == 'train':
            loss.backward()
            self.optimizer.step()
        return loss, out

    def train(self,
              train_loader,
              valid_loader,
              max_epochs=100,
              file_path=None,
              best_valid_loss=float("Inf"),
              eval_threshold=0.5,
              early_stopping_tolerance=5,
              early_stopping_min_delta=0):

        # print(self.model)
        # initialize running values
        running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        # training loop
        self.model.train()
        self.model.to(self.device)
        epoch_progress = tqdm(range(max_epochs), position=1, leave=True)
        tqdm_train_loader = tqdm(train_loader, position=0, leave=True)
        for epoch in range(max_epochs):

            for batch in train_loader:
                loss, _ = self.full_pass_step(batch, mode='train')
                # update running values
                running_loss += loss.item()
                global_step += 1

                tqdm_train_loader.update()

            # evaluation
            valid_running_loss, eval_report = self.evaluate(valid_loader)
            average_train_loss = running_loss / len(train_loader)
            average_valid_loss = valid_running_loss / len(valid_loader)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            global_steps_list.append(global_step)

            # print progress
            precision, recall, f1 = eval_report['weighted avg']['precision'], eval_report['weighted avg']['recall'], \
                                    eval_report['weighted avg']['f1-score']
            print(
                'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, precision: {:.2f}, recall: {:.2f}, F1: {:.2f}'
                .format(epoch + 1, max_epochs, global_step, max_epochs * len(train_loader),
                        average_train_loss, average_valid_loss, precision, recall, f1))
            # checkpoint
            # if best_valid_loss > average_valid_loss:
            #     best_valid_loss = average_valid_loss
            #     save_checkpoint(file_path, '/model.pt', self.model, self.optimizer, best_valid_loss)
            #     save_metrics(file_path, '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
            epoch_progress.update()
            tqdm_train_loader.reset()

            # resetting running values
            # early_stopper(average_train_loss, average_valid_loss)
            self.model.train()
            running_loss = 0.0
            # if early_stopper.early_stop:
            #     print(f'Early stop - Trained for [{epoch}/{max_epochs}] epochs')
            #     break

        tqdm_train_loader.close()
        epoch_progress.close()
        # save_metrics(file_path, '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
        print('Finished Training!')

    def evaluate(self, test_loader, ignore_vague=False, print_report=False):
        test_running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for test_batch in test_loader:
                test_loss, output = self.full_pass_step(test_batch, mode='eval')
                test_running_loss += test_loss.item()
                predictions += [self.id_2_label[idx] for idx in torch.argmax(output, dim=1).tolist()]
                true_labels += test_batch['label']

            if ignore_vague:
                true_labels = [p_l if t_l == 'VAGUE' else t_l for t_l, p_l in zip(true_labels, predictions)]
            report = classification_report(true_labels, predictions, zero_division=True, output_dict=True)
            if print_report:
                print(classification_report(true_labels, predictions, zero_division=True))
        return test_running_loss, report
