import torch

class Trainer:
    def init(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        optimizer,
        scheduler,
        batch_size=32,
        num_epochs=100,
        clip_grad=5,
        learning_rate=0.01,
        max_early_stop_epochs: 3,
        model_dir: 'model',
        stop_early_thresh: 0.001,
        thresh=0.8

    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_grad = clip_grad
        self.learning_rate = learning_rate
        self.epoch_index = 0
        self.stop_early_thresh = stop_early_thresh
        self.max_early_stop_epochs = max_early_stop_epochs
        self.num_early_stop_epochs = 0
        self.thresh = thresh
        self.stop_early = False

        self.state = {
           'loss_state': {'train': [], 'val': [],'best': 1e8},
           'f1_state': {'train': [], 'val': [],'best': 0.0},
           'accuracy_state': {'train': [], 'val': [],'best': 0.0},
        }

    def train():
        if os.path.exists(self.model_dir) is False:
            os.mkdir(self.model_dir)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("device: ", device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.model.to(device)

        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        try:
            for epoch in range(self.num_epochs):
                start_time = time.time()
                print("Epoch: ", epoch + 1)
                self.epoch_index = epoch
                train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
                train_loss = 0.
                self.model.train()
                for i, (x_vector, y_vector, x_mask) in enumerate(train_loader):
                    optimizer.zero_grad()
                    loss = model.forward_loss(x_vector.to(device), y_vector.to(device), x_mask.to(device))
                    loss.backward()

                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)

                    optimizer.step()
                    train_loss += (loss.item() - train_loss) / (i + 1)
                    # if (i + 1) % 10 == 0:
                    #     print("\tStep: {} train_loss: {}".format(i + 1, loss.item()))

                # update train loss state
                self.loss_state['train'].append(train_loss)

                self.optimizer.zero_grad()

                # validate model
                val_loss, f1, acc = self.model.validate(val_loader, thresh=self.thresh, device=device)

                # scheduler.step(val_loss)
                self.scheduler.step()

                self.loss_state['val'].append(val_loss)
                self.f1_state['val'].append(val_loss)
                self.accuracy_state['val'].append(val_loss)

                ## update valadation state
                print('*'*50 + '*********')

                self.update_state()

                if self.stop_early:
                    print('Stop Early...!')
                    print('\nSave last model at {}'.format(self.model_dir + '/final_model.pth'))
                    torch.save(model.state_dict(), self.model_dir + '/final_model.pth')
                    evaluate(self.model, self.test_dataset)
                    break

            return self.model    

        except KeyboardInterrupt:
            print('\nSave last model at {}'.format(args.model_dir + '/final_model.pth'))
            torch.save(model.state_dict(), args.model_dir + '/final_model.pth')
            evaluate(self.model, self.test_dataset)


    def evaluate(self, test_dataset):
        model.eval()
        y_pred = []
        y_true = []

        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

        for i, (x_vector, y_vector, x_mask) in enumerate(test_loader):
            out = model(x_vector.to(device), x_mask.to(device))
            y_pred.append((out > args.thresh).long())
            y_true.append(y_vector.long())


        y_true = torch.cat(y_true, dim=-1).cpu().detach().numpy()
        y_pred = torch.cat(y_pred, dim=-1).cpu().detach().numpy()

        acc, sub_acc, f1, precision, recall, hamming_loss = get_multi_label_metrics(y_true=y_true, y_pred=y_pred)
        print()
        print('*'*70 + '*********')
        print('*'*37 + "EVALUATION" + '*'*35)
        print()
        print('f1: {}    precision: {}    recall: {}'.format(f1, precision, recall))
        print('accuracy: {}    sub accuracy : {}    hamming loss: {}'.format(acc, sub_acc, hamming_loss))
        print()
        print('*'*70 + '*********')
        print('*'*70 + '*********')

    def update_state(self, key_metrics='loss'):
        # Save last checkpoint
        torch.save(self.state_dict(), self.model_dir + '/last_checkpoint.pth')

        # Save epoch checkpoint
        if save_all_checkpoint:
            torch.save(model.state_dict(), self.model_dir + '/checkpoint{}.pth'.format(self.epoch_index))
        
        metrics_t = self.state[key_metrics]['val']
        best_metric = self.state[key_metrics]['best']
        # save best model
        if metrics_t < best_metric:
            torch.save(self.state_dict(), self.model_dir + '/best_model.pth')
            self.state[key_metrics] = metrics_t
            self.num_early_stop_epochs = 0
        else:
            self.num_early_stop_epochs += 1

        if self.num_early_stop_epochs >= self.max_early_stop_epochs:
            self.stop_early = True

    def save_checkpoint(self, model_file: Union[str, Path]):
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        test_dataset = self.test_dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        torch.save(self, str(model_file), pickle_protocol=4)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @classmethod
    def load_checkpoint(cls, checkpoint: Union[Path, str], train_dataset, val_dataset, test_dataset):
        trainer = torch.load(checkpoint)
        trainer.train_dataset = train_dataset
        trainer.val_dataset = val_dataset
        trainer.test_dataset = test_dataset
        return model