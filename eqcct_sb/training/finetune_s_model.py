"""
Fine tuning the S model 
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import seisbench.data as sbd
import seisbench.generate as sbg 
from torch.utils.data import DataLoader, Subset
from eqcct_sb.models.predictor_pt import EQCCTModelS
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

def get_data_source(ds_choice=1, smple_rate=100): 
    if ds_choice == 0: 
        data = sbd.STEAD(sampling_rate=smple_rate, component_order="ZNE")
        print(f"Using STEAD for evaluation...")

    elif ds_choice == 1: 
        data = sbd.TXED(sampling_rate=smple_rate, component_order="ZNE")
        print(f"Using TXED for evaluation...")

    phase_dict = {
        "trace_s_arrival_sample": "S"
    }

    augmentations = [sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=2000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, model_labels='S', dim=0, sigma=10, shape='gaussian'), 
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak", key=("X", "X")),
                    sbg.ChangeDtype(np.float32)]

    return data, augmentations 

    
def prepare_inputs(batch, device):
    waveforms = batch["X"].to(device)  # shape: [B, 3, 6000]
    true_picks = batch["y"].to(torch.float32).to(device)  # shape: [B, 6000, 1] or [B, 6000]

    if true_picks.ndim == 3:
        true_picks = true_picks[:, 0, :]  # Channel 0 is 'S'

    if true_picks.ndim == 2:
        true_picks = true_picks.unsqueeze(-1)  # → [B, 6000, 1]


    waveforms = waveforms.permute(0, 2, 1)  # [B, 6000, 3]
    return waveforms, waveforms, true_picks

class SimpleTrainer(): 
    def __init__(self, model, optimizer, criterion, device):
        self.model = model 
        self.optimizer = optimizer
        self.criterion = criterion 
        self.device = device 

    def train_loop(self, loader, epoch, threshold=0.4, print_every=200, writer=None): 
        self.model.train()
        # all_preds = []
        # all_labels = [] 
        total_tp = total_fp = total_fn = running_loss = 0.0 

        for batch_idx, batch in enumerate(loader):
            waveforms, inputs, true_picks = prepare_inputs(batch, self.device)
            outputs = self.model(inputs) # Forward pass inputs to model
            loss = self.criterion(outputs, true_picks) 
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()

            # Compute batch-level F1
            running_loss += loss.item()
            bin_preds = (outputs > threshold).to(torch.int32)# .cpu().numpy().ravel()
            bin_labels = (true_picks > threshold).to(torch.int32)#.cpu().numpy().ravel()

            tp = (bin_preds * bin_labels).sum().item()
            fp = (bin_preds * (1 - bin_labels)).sum().item()
            fn = ((1 - bin_preds) * bin_labels).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn


            # all_preds.append(preds_np)
            # all_labels.append(labels_np)
            
            # # Print intermediate metrics
            # if (batch_idx + 1) % print_every == 0:
            #     preds_so_far = np.concatenate(all_preds)
            #     labels_so_far = np.concatenate(all_labels) 
            #     precision = precision_score(labels_so_far, preds_so_far, zero_division=0)
            #     recall = recall_score(labels_so_far, preds_so_far, zero_division=0)
            #     f1 = f1_score(labels_so_far, preds_so_far, zero_division=0)
            #     avg_loss = running_loss / print_every
            #     print(f"[Epoch {epoch+1} | Batch {batch_idx+1}] Loss: {avg_loss:.6f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            #     if writer: 
            #         global_step = epoch * len(loader) + batch_idx
            #         writer.add_scalar("Train/Loss", avg_loss, global_step)
            #         writer.add_scalar("Train/Precision", precision, global_step)
            #         writer.add_scalar("Train/Recall", recall, global_step)
            #         writer.add_scalar("Train/F1", f1, global_step)

        # all_preds = np.concatenate(all_preds)
        # all_labels = np.concatenate(all_labels)

        precision = total_tp / (total_tp + total_fp + 1e-8) # precision_score(all_labels, all_preds, zero_division=0)
        recall = total_tp / (total_tp + total_fn + 1e-8) # recall_score(all_labels, all_preds, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall + 1e-8) # f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = running_loss / len(loader)
        return precision, recall, f1, avg_loss


    def evaluate(self, loader, threshold=0.4, print_every=200, writer=None):
        self.model.eval()
        # all_preds = []
        # all_labels = []
        total_tp = total_fp = total_fn = running_loss = 0.0  

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader): 
                waveforms, inputs, true_picks =  prepare_inputs(batch, self.device)
                outputs = self.model(inputs)
                # Loss calculation
                loss = self.criterion(outputs, true_picks)
                running_loss += loss.item()
                # Label binarization 
                bin_preds = (outputs > threshold).to(torch.int32)# .cpu().numpy().ravel()
                bin_labels = (true_picks > threshold).to(torch.int32)#.cpu().numpy().ravel()

                tp = (bin_preds * bin_labels).sum().item()
                fp = (bin_preds * (1 - bin_labels)).sum().item()
                fn = ((1 - bin_preds) * bin_labels).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn
                # all_preds.append(preds_np)
                # all_labels.append(labels_np)
                
            
        # all_preds = np.concatenate(all_preds)
        # all_labels = np.concatenate(all_labels)

        precision = total_tp / (total_tp + total_fp + 1e-8) # precision_score(all_labels, all_preds, zero_division=0)
        recall = total_tp / (total_tp + total_fn + 1e-8) # recall_score(all_labels, all_preds, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall + 1e-8) # f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = running_loss / len(loader)

        if writer: 
            writer.add_scalar(f"Val/F1_thresh_{threshold:.2f}", f1)
            writer.add_scalar(f"Val/Loss_thresh_{threshold:.2f}", avg_loss)
        return precision, recall, f1, avg_loss


from sklearn.metrics import f1_score
from tqdm import tqdm
def finetune_full_s_model(patience=3, thresh_grid=None, max_epochs=20, ds_choice=1, smple_rate=100, smodel=None, best_model=None, final_model=None):

    if thresh_grid is None:
        thresh_grid = np.linspace(0.1, 0.9, 17)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/finetune_eqcct_s")


    # Load data 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_source, augmentations = get_data_source(ds_choice, smple_rate) 
    train_data, val_data, _ = data_source.train_dev_test()
    # print(f"using {len(train_data)} samples for finetuning")

    # Training Generator
    train_generator = sbg.GenericGenerator(train_data)
    train_generator.add_augmentations(augmentations)
    train_loader = DataLoader(train_generator, batch_size=16, shuffle=True)

    # Validation Generator
    val_generator = sbg.GenericGenerator(val_data)
    val_generator.add_augmentations(augmentations)
    val_loader = DataLoader(val_generator, batch_size=16, shuffle=False)

    # Load model 
    model = EQCCTModelS().to(device)
    model.load_state_dict(torch.load(smodel))
    # model = torch.compile(model)
    for param in model.parameters():
        param.requires_grad = True

    # Optimizers, loss, Trainer
    criterion = nn.BCELoss() # Binary Classification Loss (diff. between predicted probability distribution and accurate binary labels of the data)
    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    trainer = SimpleTrainer(model, optimizer, criterion, device)

    # Training loop (threshold logic removed)
    best_val_f1 = 0.0
    best_epoch = -1 
    best_thresh = None 
    no_improve = 0 

    print("Beginning Training Loop...")
    for epoch in range(max_epochs):
        # 1) Train
        train_prec, train_rec, train_f1, train_loss = trainer.train_loop(
            train_loader,
            epoch,
            threshold=0.4,
            print_every=1000,
            writer=writer
        )
        print(f"[Epoch {epoch+1}] Avg Train Loss: {train_loss:.6f}, Train F1 @0.40: {train_f1:.4f}, Train Prec: {train_prec:.4f}, Train Rec: {train_rec:.4f}")
        writer.add_scalar("Train/Epoch_Loss", train_loss, epoch)

        # 2) Validate once per epoch

        # 2) Get raw val outputs & labels (once)
        # model.eval()
        # all_preds = []
        # all_labels = []
        # print(f"Epoch {epoch+1}: running validation on {len(val_loader)} batches…")
        # with torch.no_grad():
        #     for i, batch in enumerate(tqdm(val_loader, desc="Val")):
        #         if (i+1) % 100 == 0:
        #             print(f"  val batch {i+1}/{len(val_loader)}")
        #         wave, inp, true = prepare_inputs(batch, device)
        #         out = model(inp).squeeze(-1).cpu().numpy().ravel()
        #         lbl = true.squeeze(-1).cpu().numpy().ravel()
        #         all_preds.append(out)
        #         all_labels.append(lbl)
        # all_preds = np.concatenate(all_preds)
        # all_labels = np.concatenate(all_labels)

        # true_bin = (all_labels > 0.5).astype(int)
        # # 3) Sweep threshold grid for best F1
        # val_f1s = []
        # for t in thresh_grid:
        #     print(f"Epoch {epoch+1}: validation threshold of {t}")
        #     bpred = (all_preds > t).astype(int)
        #     f1 = f1_score(true_bin, bpred, zero_division=0)
        #     val_f1s.append(f1)
        # best_i = int(np.argmax(val_f1s))
        # epoch_thresh = float(thresh_grid[best_i])
        # epoch_val_f1 = val_f1s[best_i]
        # epoch_val_loss = np.mean(nn.BCELoss(reduction='none')(
        #     torch.tensor(all_preds), torch.tensor(all_labels)
        # ).numpy())

        # print(f"[Epoch {epoch+1}] → Best Val-Thresh: {epoch_thresh:.2f}, Val-F1: {epoch_val_f1:.4f}, Val-Loss: {epoch_val_loss:.6f}")
        # writer.add_scalar("Val/BestThresh", epoch_thresh, epoch)
        # writer.add_scalar("Val/F1", epoch_val_f1, epoch)
        # writer.add_scalar("Val/Loss", epoch_val_loss, epoch)

        val_prec, val_rec, val_f1, val_loss = trainer.evaluate(val_loader, threshold=0.4, print_every=1000, writer=writer)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}, Val F1 @0.40: {val_f1:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}")
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/F1@0.40", val_f1, epoch)

        # 4) Early-stop check
        # if epoch_val_f1 > best_val_f1:
        if val_f1 > best_val_f1: 
            best_val_f1 = val_f1 # epoch_val_f1
            # best_thresh = epoch_thresh
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model)
            print(f"  ↳ New best (F1: {best_val_f1:.4f}) saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs (best F1={best_val_f1:.4f} at epoch {best_epoch+1}). Stopping.")
                break

    # final checkpoint
    torch.save(model.state_dict(), final_model)
    writer.close()

    print(f"Training completed. Best Val-F1={best_val_f1:.4f}.")