# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import time
import pdb
import sys

import numpy as np
import tensorflow as tf

from baskerville import metrics


def parse_loss(
    loss_label,
    strategy=None,
    keras_fit: bool = True,
    spec_weight: float = 1,
    total_weight: float = 1,
):
    """Parse loss function from label, strategy, and fitting method.

    Args:
      loss_label (str): Loss function label.
      strategy: tf.distribute.Strategy object.
      keras_fit (bool): Use Keras fit method instead of custom loop.
      spec_weight (float): Specificity weight for PoissonKL.
      total_weight (float): Total weight for PoissionMultinomial.

    Returns:
      loss_fn: tf.keras.losses.Loss object.
    """
    if strategy is not None and not keras_fit:
        if loss_label == "mse":
            loss_fn = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            )
        elif loss_label == "bce":
            loss_fn = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )
        elif loss_label == "mlm":
            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )
        elif loss_label == "poisson_mn":
            loss_fn = metrics.PoissonMultinomial(
                total_weight, reduction=tf.keras.losses.Reduction.NONE
            )
        else:
            loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    else:
        if loss_label == "mse":
            loss_fn = tf.keras.losses.MeanSquaredError()
        elif loss_label == "mse_udot":
            loss_fn = metrics.MeanSquaredErrorUDot(spec_weight)
        elif loss_label == "bce":
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif loss_label == "mlm":
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
        elif loss_label == "poisson_kl":
            loss_fn = metrics.PoissonKL(spec_weight)
        elif loss_label == "poisson_mn":
            loss_fn = metrics.PoissonMultinomial(total_weight)
        else:
            loss_fn = tf.keras.losses.Poisson()

    return loss_fn


class Trainer:
    """Model training class.

    Args:
      params (dict): Training parameters dictionary.
      train_data: Dataset object or list of Dataset objects.
      eval_data: Dataset object or list of Dataset objects.
      out_dir (str): Output directory name.
      strategy: tf.distribute.Strategy object.
      num_gpu (int): Number of GPUs to use. Default: 1.
      keras_fit (bool): Use Keras fit method instead of custom loop.
    """

    def __init__(
        self,
        params: dict,
        train_data,
        eval_data,
        out_dir: str,
        strategy=None,
        num_gpu: int = 1,
        keras_fit: bool = False,
    ):
        self.params = params
        self.train_data = train_data
        if type(self.train_data) is not list:
            self.train_data = [self.train_data]
        self.eval_data = eval_data
        if type(self.eval_data) is not list:
            self.eval_data = [self.eval_data]
        self.out_dir = out_dir
        self.strategy = strategy
        self.num_gpu = num_gpu
        self.batch_size = self.train_data[0].batch_size
        self.compiled = False
        
        #mlm mask rate
        self.mask_rate = self.params.get("mask_rate", 0)
        
        #other mlm arguments
        self.exon_loss_scale = self.params.get("exon_loss_scale", None)
        self.non_exon_loss_scale = self.params.get("non_exon_loss_scale", None)
        self.repeat_loss_scale = self.params.get("repeat_loss_scale", None)
        self.non_repeat_loss_scale = self.params.get("non_repeat_loss_scale", None)
        self.exon_mut_rate = self.params.get("exon_mut_rate", None)
        self.repeat_eval = self.params.get("repeat_eval", 1)

        # early stopping
        self.patience = self.params.get("patience", 20)

        # compute batches/epoch
        self.train_epoch_batches = [td.batches_per_epoch() for td in self.train_data]
        self.eval_epoch_batches = [ed.batches_per_epoch() for ed in self.eval_data]
        self.train_epochs_min = self.params.get("train_epochs_min", 1)
        self.train_epochs_max = self.params.get("train_epochs_max", 10000)
        self.steps_per_epoch_max = self.params.get("steps_per_epoch_max", None)
        self.shuffle_records = self.params.get("shuffle_records", False)

        # dataset
        self.num_datasets = len(self.train_data)
        self.dataset_indexes = []
        for di in range(self.num_datasets):
            self.dataset_indexes += [di] * self.train_epoch_batches[di]
        self.dataset_indexes = np.array(self.dataset_indexes)

        # loss
        self.spec_weight = self.params.get("spec_weight", 1)
        self.total_weight = self.params.get("total_weight", 1)
        self.loss = self.params.get("loss", "poisson").lower()
        self.loss_fn = parse_loss(
            self.loss, self.strategy, keras_fit, self.spec_weight, self.total_weight
        )

        # optimizer
        self.make_optimizer()

    def compile(self, seqnn_model):
        for model in seqnn_model.models:
            if self.loss == "bce":
                model_metrics = [
                    metrics.SeqAUC(curve="ROC"),
                    metrics.SeqAUC(curve="PR"),
                ]
            elif self.loss == "mlm":
                model_metrics = []
            else:
                num_targets = model.output_shape[-1]
                model_metrics = [metrics.PearsonR(num_targets), metrics.R2(num_targets)]

            model.compile(
                loss=self.loss_fn, optimizer=self.optimizer, metrics=model_metrics
            )
        self.compiled = True

    def fit_keras(self, seqnn_model):
        if not self.compiled:
            self.compile(seqnn_model)

        if self.loss in ["bce", "mlm"]:
            early_stop = EarlyStoppingMin(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=self.patience,
                min_epoch=self.train_epochs_min,
            )
            save_best = tf.keras.callbacks.ModelCheckpoint(
                "%s/model_best.h5" % self.out_dir,
                save_best_only=True,
                mode="min",
                monitor="val_loss",
                verbose=1,
            )
        else:
            early_stop = EarlyStoppingMin(
                monitor="val_pearsonr",
                mode="max",
                verbose=1,
                patience=self.patience,
                min_epoch=self.train_epochs_min,
            )
            save_best = tf.keras.callbacks.ModelCheckpoint(
                "%s/model_best.h5" % self.out_dir,
                save_best_only=True,
                mode="max",
                monitor="val_pearsonr",
                verbose=1,
            )

        callbacks = [
            early_stop,
            tf.keras.callbacks.TensorBoard(self.out_dir),
            tf.keras.callbacks.ModelCheckpoint("%s/model_check.h5" % self.out_dir),
            save_best,
        ]

        seqnn_model.model.fit(
            self.train_data[0].dataset,
            epochs=self.train_epochs_max,
            steps_per_epoch=self.train_epoch_batches[0],
            callbacks=callbacks,
            validation_data=self.eval_data[0].dataset,
            validation_steps=self.eval_epoch_batches[0],
        )

    def fit2(self, seqnn_model):
        """Train the model using a custom loop for two separate datasets."""
        if not self.compiled:
            self.compile(seqnn_model)

        assert len(seqnn_model.models) >= self.num_datasets

        # inform optimizer about all trainable variables (v2.11-)
        vars_set = set()
        trainable_vars = []
        for di in range(self.num_datasets):
            for v in seqnn_model.models[di].trainable_variables:
                if v.name not in vars_set:
                    vars_set.add(v.name)
                    trainable_vars.append(v)
        try:
            self.optimizer.build(trainable_vars)
        except AttributeError:
            pass

        ################################################################
        # prep

        # metrics
        train_loss, train_r, train_r2 = [], [], []
        valid_loss, valid_r, valid_r2 = [], [], []
        for di in range(self.num_datasets):
            num_targets = seqnn_model.models[di].output_shape[-1]
            train_loss.append(tf.keras.metrics.Mean(name="train%d_loss" % di))
            train_r.append(metrics.PearsonR(num_targets, name="train%d_r" % di))
            train_r2.append(metrics.R2(num_targets, name="train%d_r2" % di))
            valid_loss.append(tf.keras.metrics.Mean(name="valid%d_loss" % di))
            valid_r.append(metrics.PearsonR(num_targets, name="valid%d_r" % di))
            valid_r2.append(metrics.R2(num_targets, name="valid%d_r2" % di))

        if self.strategy is None:
            # generate decorated train steps
            @tf.function
            def train_step0(x, y):
                with tf.GradientTape() as tape:
                    pred = seqnn_model.models[0](x, training=True)
                    loss = self.loss_fn(y, pred) + sum(seqnn_model.models[0].losses)
                train_loss[0](loss)
                train_r[0](y, pred)
                train_r2[0](y, pred)
                gradients = tape.gradient(
                    loss, seqnn_model.models[0].trainable_variables
                )
                self.optimizer.apply_gradients(
                    zip(gradients, seqnn_model.models[0].trainable_variables)
                )

            @tf.function
            def eval_step0(x, y):
                pred = seqnn_model.models[0](x, training=False)
                loss = self.loss_fn(y, pred) + sum(seqnn_model.models[0].losses)
                valid_loss[0](loss)
                valid_r[0](y, pred)
                valid_r2[0](y, pred)

            if self.num_datasets > 1:

                @tf.function
                def train_step1(x, y):
                    with tf.GradientTape() as tape:
                        pred = seqnn_model.models[1](x, training=True)
                        loss = self.loss_fn(y, pred) + sum(seqnn_model.models[1].losses)
                    train_loss[1](loss)
                    train_r[1](y, pred)
                    train_r2[1](y, pred)
                    gradients = tape.gradient(
                        loss, seqnn_model.models[1].trainable_variables
                    )
                    self.optimizer.apply_gradients(
                        zip(gradients, seqnn_model.models[1].trainable_variables)
                    )

                @tf.function
                def eval_step1(x, y):
                    pred = seqnn_model.models[1](x, training=False)
                    loss = self.loss_fn(y, pred) + sum(seqnn_model.models[1].losses)
                    valid_loss[1](loss)
                    valid_r[1](y, pred)
                    valid_r2[1](y, pred)

        else:

            def train_step0(x, y):
                with tf.GradientTape() as tape:
                    pred = seqnn_model.models[0](x, training=True)
                    loss_batch_len = self.loss_fn(y, pred)
                    loss_batch = tf.reduce_mean(loss_batch_len, axis=-1)
                    loss = tf.reduce_sum(loss_batch) / self.batch_size
                    loss += sum(seqnn_model.models[0].losses) / self.num_gpu
                train_r[0](y, pred)
                train_r2[0](y, pred)
                gradients = tape.gradient(
                    loss, seqnn_model.models[0].trainable_variables
                )
                self.optimizer.apply_gradients(
                    zip(gradients, seqnn_model.models[0].trainable_variables)
                )
                return loss

            @tf.function
            def train_step0_distr(xd, yd):
                replica_losses = self.strategy.run(train_step0, args=(xd, yd))
                loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, replica_losses, axis=None
                )
                train_loss[0](loss)

            def eval_step0(x, y):
                pred = seqnn_model.models[0](x, training=False)
                loss = self.loss_fn(y, pred) + sum(seqnn_model.models[0].losses)
                valid_loss[0](loss)
                valid_r[0](y, pred)
                valid_r2[0](y, pred)

            @tf.function
            def eval_step0_distr(xd, yd):
                return self.strategy.run(eval_step0, args=(xd, yd))

            if self.num_datasets > 1:

                def train_step1(x, y):
                    with tf.GradientTape() as tape:
                        pred = seqnn_model.models[1](x, training=True)
                        loss_batch_len = self.loss_fn(y, pred)
                        loss_batch = tf.reduce_mean(loss_batch_len, axis=-1)
                        loss = tf.reduce_sum(loss_batch) / self.batch_size
                        loss += sum(seqnn_model.models[1].losses) / self.num_gpu
                    train_loss[1](loss)
                    train_r[1](y, pred)
                    train_r2[1](y, pred)
                    gradients = tape.gradient(
                        loss, seqnn_model.models[1].trainable_variables
                    )
                    self.optimizer.apply_gradients(
                        zip(gradients, seqnn_model.models[1].trainable_variables)
                    )
                    return loss

                @tf.function
                def train_step1_distr(xd, yd):
                    replica_losses = self.strategy.run(train_step1, args=(xd, yd))
                    loss = self.strategy.reduce(
                        tf.distribute.ReduceOp.SUM, replica_losses, axis=None
                    )
                    train_loss[1](loss)

                def eval_step1(x, y):
                    pred = seqnn_model.models[1](x, training=False)
                    loss = self.loss_fn(y, pred) + sum(seqnn_model.models[1].losses)
                    valid_loss[1](loss)
                    valid_r[1](y, pred)
                    valid_r2[1](y, pred)

                @tf.function
                def eval_step1_distr(xd, yd):
                    return self.strategy.run(eval_step1, args=(xd, yd))

        # checkpoint manager
        managers = []
        for di in range(self.num_datasets):
            ckpt = tf.train.Checkpoint(
                model=seqnn_model.models[di], optimizer=self.optimizer
            )
            ckpt_dir = "%s/model%d" % (self.out_dir, di)
            manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
            if manager.latest_checkpoint:
                ckpt.restore(manager.latest_checkpoint)
                ckpt_end = 5 + manager.latest_checkpoint.find("ckpt-")
                epoch_start = int(manager.latest_checkpoint[ckpt_end:])
                if self.strategy is None:
                    opt_iters = self.optimizer.iterations
                else:
                    opt_iters = self.optimizer.iterations.values[0]
                print(
                    "Checkpoint restored at epoch %d, optimizer iteration %d."
                    % (epoch_start, opt_iters)
                )
            else:
                print("No checkpoints found.")
                epoch_start = 0
            managers.append(manager)

        # improvement variables
        valid_best = [-np.inf] * self.num_datasets
        unimproved = [0] * self.num_datasets

        ################################################################
        # training loop

        first_step = True
        for ei in range(epoch_start, self.train_epochs_max):
            if ei >= self.train_epochs_min and np.min(unimproved) > self.patience:
                break
            else:
                # shuffle datasets
                np.random.shuffle(self.dataset_indexes)

                # get iterators
                train_data_iters = [iter(td.dataset) for td in self.train_data]

                # train
                t0 = time.time()
                for di in self.dataset_indexes:
                    x, y = safe_next(train_data_iters[di])
                    if self.strategy is None:
                        if di == 0:
                            train_step0(x, y)
                        else:
                            train_step1(x, y)
                    else:
                        if di == 0:
                            train_step0_distr(x, y)
                        else:
                            train_step1_distr(x, y)
                    if first_step:
                        print("Successful first step!", flush=True)
                        first_step = False

                print("Epoch %d - %ds" % (ei, (time.time() - t0)))
                for di in range(self.num_datasets):
                    print("  Data %d" % di, end="")
                    model = seqnn_model.models[di]

                    # print training accuracy
                    print(
                        " - train_loss: %.4f" % train_loss[di].result().numpy(), end=""
                    )
                    print(" - train_r: %.4f" % train_r[di].result().numpy(), end="")
                    print(" - train_r: %.4f" % train_r2[di].result().numpy(), end="")

                    # evaluate
                    for x, y in self.eval_data[di].dataset:
                        if self.strategy is None:
                            if di == 0:
                                eval_step0(x, y)
                            else:
                                eval_step1(x, y)
                        else:
                            if di == 0:
                                eval_step0_distr(x, y)
                            else:
                                eval_step1_distr(x, y)

                    # print validation accuracy
                    print(
                        " - valid_loss: %.4f" % valid_loss[di].result().numpy(), end=""
                    )
                    print(" - valid_r: %.4f" % valid_r[di].result().numpy(), end="")
                    print(" - valid_r2: %.4f" % valid_r2[di].result().numpy(), end="")
                    early_stop_stat = valid_r[di].result().numpy()

                    # checkpoint
                    managers[di].save()
                    model.save(
                        "%s/model%d_check.h5" % (self.out_dir, di),
                        include_optimizer=False,
                    )

                    # check best
                    if early_stop_stat > valid_best[di]:
                        print(" - best!", end="")
                        unimproved[di] = 0
                        valid_best[di] = early_stop_stat
                        model.save(
                            "%s/model%d_best.h5" % (self.out_dir, di),
                            include_optimizer=False,
                        )
                    else:
                        unimproved[di] += 1
                    print("", flush=True)

                    # reset metrics
                    train_loss[di].reset_states()
                    train_r[di].reset_states()
                    train_r2[di].reset_states()
                    valid_loss[di].reset_states()
                    valid_r[di].reset_states()
                    valid_r2[di].reset_states()

    def fit_tape(self, seqnn_model):
        """Train the model using a custom tf.GradientTape loop."""
        if not self.compiled:
            self.compile(seqnn_model)
        model = seqnn_model.model

        # metrics
        num_targets = model.output_shape[-1]
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_r = metrics.PearsonR(num_targets, name="train_r")
        train_r2 = metrics.R2(num_targets, name="train_r2")
        valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        valid_r = metrics.PearsonR(num_targets, name="valid_r")
        valid_r2 = metrics.R2(num_targets, name="valid_r2")

        if self.strategy is None:

            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss = self.loss_fn(y, pred) + sum(model.losses)
                train_loss(loss)
                train_r(y, pred)
                train_r2(y, pred)
                gradients = tape.gradient(loss, model.trainable_variables)
                if self.agc_clip is not None:
                    gradients = adaptive_clip_grad(
                        model.trainable_variables, gradients, self.agc_clip
                    )
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )

            @tf.function
            def eval_step(x, y):
                pred = model(x, training=False)
                loss = self.loss_fn(y, pred) + sum(model.losses)
                valid_loss(loss)
                valid_r(y, pred)
                valid_r2(y, pred)

        else:

            def train_step(x, y):
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss_batch_len = self.loss_fn(y, pred)
                    loss_batch = tf.reduce_mean(loss_batch_len, axis=-1)
                    loss = tf.reduce_sum(loss_batch) / self.batch_size
                    loss += sum(model.losses) / self.num_gpu
                train_r(y, pred)
                train_r2(y, pred)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )
                return loss

            @tf.function
            def train_step_distr(xd, yd):
                replica_losses = self.strategy.run(train_step, args=(xd, yd))
                loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, replica_losses, axis=None
                )
                train_loss(loss)

            def eval_step(x, y):
                pred = model(x, training=False)
                loss = self.loss_fn(y, pred) + sum(model.losses)
                valid_loss(loss)
                valid_r(y, pred)
                valid_r2(y, pred)

            @tf.function
            def eval_step_distr(xd, yd):
                return self.strategy.run(eval_step, args=(xd, yd))

        # checkpoint manager
        ckpt = tf.train.Checkpoint(model=seqnn_model.model, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, self.out_dir, max_to_keep=1)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            ckpt_end = 5 + manager.latest_checkpoint.find("ckpt-")
            epoch_start = int(manager.latest_checkpoint[ckpt_end:])
            if self.strategy is None:
                opt_iters = self.optimizer.iterations
            else:
                opt_iters = self.optimizer.iterations.values[0]
            print(
                "Checkpoint restored at epoch %d, optimizer iteration %d."
                % (epoch_start, opt_iters)
            )
        else:
            print("No checkpoints found.")
            epoch_start = 0

        # improvement variables
        valid_best = -np.inf
        unimproved = 0

        # training loop
        for ei in range(epoch_start, self.train_epochs_max):
            if ei >= self.train_epochs_min and unimproved > self.patience:
                break
            else:
                # train
                t0 = time.time()
                train_iter = iter(self.train_data[0].dataset)
                for si in range(self.train_epoch_batches[0]):
                    x, y = safe_next(train_iter)
                    if self.strategy is not None:
                        train_step_distr(x, y)
                    else:
                        train_step(x, y)
                    if ei == epoch_start and si == 0:
                        print("Successful first step!", flush=True)

                # evaluate
                for x, y in self.eval_data[0].dataset:
                    if self.strategy is not None:
                        eval_step_distr(x, y)
                    else:
                        eval_step(x, y)

                # print training accuracy
                train_loss_epoch = train_loss.result().numpy()
                train_r_epoch = train_r.result().numpy()
                train_r2_epoch = train_r2.result().numpy()
                print(
                    "Epoch %d - %ds - train_loss: %.4f - train_r: %.4f - train_r2: %.4f"
                    % (
                        ei,
                        (time.time() - t0),
                        train_loss_epoch,
                        train_r_epoch,
                        train_r2_epoch,
                    ),
                    end="",
                )

                # print validation accuracy
                valid_loss_epoch = valid_loss.result().numpy()
                valid_r_epoch = valid_r.result().numpy()
                valid_r2_epoch = valid_r2.result().numpy()
                print(
                    " - valid_loss: %.4f - valid_r: %.4f - valid_r2: %.4f"
                    % (valid_loss_epoch, valid_r_epoch, valid_r2_epoch),
                    end="",
                )

                # checkpoint
                manager.save()
                seqnn_model.save("%s/model_check.h5" % self.out_dir)

                # check best
                valid_best_epoch = valid_r_epoch + valid_r2_epoch / 4
                if valid_best_epoch > valid_best:
                    print(" - best!", end="")
                    unimproved = 0
                    valid_best = valid_best_epoch
                    seqnn_model.save("%s/model_best.h5" % self.out_dir)
                else:
                    unimproved += 1
                print("", flush=True)

                # reset metrics
                train_loss.reset_states()
                train_r.reset_states()
                train_r2.reset_states()
                valid_loss.reset_states()
                valid_r.reset_states()
                valid_r2.reset_states()

    def fit_mlm(self, seqnn_model):
        """Train the model in masked language modeling mode using a custom tf.GradientTape loop."""
        if not self.compiled:
            self.compile(seqnn_model)
        model = seqnn_model.model

        # metrics
        num_features = model.output_shape[-1]
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        valid_loss = tf.keras.metrics.Mean(name="valid_loss")

        if self.strategy is None:

            @tf.function
            def train_step(x_masked, x, ind, sample_weight=None):
                with tf.GradientTape() as tape:
                    x_pred = model(x_masked, training=True)
                    x_pred = tf.gather(x_pred, ind, axis=1, batch_dims=1)
                    x_true = tf.gather(x, ind, axis=1, batch_dims=1)
                    if sample_weight is not None:
                        sample_weight = tf.gather(sample_weight, ind, axis=1, batch_dims=1)
                    
                    loss = self.loss_fn(x_true, x_pred, sample_weight=sample_weight) + sum(model.losses)
                
                train_loss(loss)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                if self.agc_clip is not None:
                    gradients = adaptive_clip_grad(
                        model.trainable_variables, gradients, self.agc_clip
                    )
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )

            @tf.function
            def eval_step(x_masked, x, ind, sample_weight=None):
                x_pred = model(x_masked, training=False)
                x_pred = tf.gather(x_pred, ind, axis=1, batch_dims=1)
                x_true = tf.gather(x, ind, axis=1, batch_dims=1)
                if sample_weight is not None:
                    sample_weight = tf.gather(sample_weight, ind, axis=1, batch_dims=1)
                
                loss = self.loss_fn(x_true, x_pred, sample_weight=sample_weight) + sum(model.losses)
                
                #valid_loss(loss)
                return loss
            
            # def prep_mlm(x, label, mask_size, exon_mask=None, training=False, augment_rc=True):
            def prep_mlm(x, label, mask_size, exon_mask=None, repeat_mask=None, training=False, augment_rc=True):
                # randomly revcomp the sequence(s) if in training mode
                if training and augment_rc :
                    do_rc = tf.cast(tf.random.uniform([x.shape[0]], minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)
                    
                    x = tf.where(
                        do_rc[:, None, None],
                        tf.reverse(x, axis=[1, 2]),
                        x,
                    )
                    
                    if exon_mask is not None :
                        exon_mask = tf.where(
                            do_rc[:, None],
                            tf.reverse(exon_mask, axis=[1]),
                            exon_mask,
                        )
                    if repeat_mask is not None:
                        repeat_mask = tf.where(
                            do_rc[:, None],
                            tf.reverse(repeat_mask, axis=[1]),
                            repeat_mask,
                        )

                
                # optionally set position-specific loss weight scales from binary mask
                sw = None
                
                # exon_mask scaling
                if exon_mask is not None and self.exon_loss_scale is not None :
                    #non_exon_loss_scale = 1. + (1. - self.exon_loss_scale) * tf.math.minimum(tf.reduce_sum(exon_mask, axis=1) / (exon_mask.shape[1] - tf.reduce_sum(exon_mask, axis=1)), 16.)
                    #sw = exon_mask * self.exon_loss_scale + (1 - exon_mask) * non_exon_loss_scale[:, None]
                    sw = exon_mask * self.exon_loss_scale + (1 - exon_mask) * self.non_exon_loss_scale
                
                # repeat_mask scaling
                if repeat_mask is not None and self.repeat_loss_scale is not None:
                    repeat_sw = repeat_mask * self.repeat_loss_scale + (1 - repeat_mask) * self.non_repeat_loss_scale
                    # print("repeat_sw: ", repeat_sw)
                    if sw is None:
                        sw = repeat_sw
                    else:
                        sw *= repeat_sw
                # print("sw: ", sw)
                # print(""+1)

                # get indices for random input mask
                ind = tf.tile(tf.range(x.shape[1], dtype=tf.int32)[None, :], (x.shape[0], 1))

                ind = tf.map_fn(
                    fn=lambda t_ind: tf.random.shuffle(t_ind),
                    elems=ind,
                    fn_output_signature=tf.int32,
                )
                
                ind = ind[:, :mask_size]

                # construct binary mask
                mask = tf.map_fn(
                    fn=lambda t_ind: tf.scatter_nd(t_ind[:, None], tf.ones(t_ind.shape[0], dtype=tf.float32), [x.shape[1]]),
                    elems=ind,
                    fn_output_signature=tf.float32,
                )[..., None]
                
                # optionally and randomly mutate part of the coding regions
                if exon_mask is not None and self.exon_mut_rate is not None :
                    
                    mut_size = tf.cast(self.exon_mut_rate * x.shape[1], dtype=tf.int32)
                    
                    mut_ind = tf.tile(tf.range(x.shape[1], dtype=tf.int32)[None, :], (x.shape[0], 1))

                    # sample positions
                    mut_ind = tf.map_fn(
                        fn=lambda t_ind: tf.random.shuffle(t_ind),
                        elems=mut_ind,
                        fn_output_signature=tf.int32,
                    )

                    mut_ind = mut_ind[:, :mut_size]
                    
                    mut_mask = tf.map_fn(
                        fn=lambda t_mut_ind: tf.scatter_nd(t_mut_ind[:, None], tf.ones(t_mut_ind.shape[0], dtype=tf.float32), [x.shape[1]]),
                        elems=mut_ind,
                        fn_output_signature=tf.float32,
                    )[..., None]
                    
                    # sample and inflate mutated nucleotide identities at positions
                    x_mut = tf.map_fn(
                        fn=lambda t_mut_ind: tf.cast(tf.one_hot(tf.scatter_nd(
                            t_mut_ind[:, None],
                            1 + tf.random.uniform([t_mut_ind.shape[0]], minval=0, maxval=4, dtype=tf.int32),
                            [x.shape[1]]
                        ), 5, dtype=tf.int32)[:, 1:], dtype=tf.float32),
                        elems=mut_ind,
                        fn_output_signature=tf.float32,
                    )
                    
                    # apply mutations
                    x = x * (1 - mut_mask * exon_mask[..., None] * (1 - mask)) + x_mut * mut_mask * exon_mask[..., None] * (1 - mask)
                
                # append mask token embedding dimension
                x_w_token = tf.concat([
                    x,
                    tf.zeros((x.shape[0], x.shape[1], 1)),
                ], axis=-1)

                # template for a fully-masked input pattern
                mask_bias = tf.concat([
                    tf.zeros((x.shape[0], x.shape[1], 4), dtype=tf.float32),
                    tf.ones((x.shape[0], x.shape[1], 1), dtype=tf.float32),
                ], axis=-1)

                # apply mask
                x_masked = x_w_token * (1 - mask) + mask_bias * mask

                # broadcast and concat label to input (along channels)
                x_masked = tf.concat([
                    x_masked,
                    tf.tile(label, (1, x.shape[1], 1)),
                ], axis=-1)
                
                return x_masked, x, ind, sw
        

        # checkpoint manager
        ckpt = tf.train.Checkpoint(model=seqnn_model.model, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, self.out_dir, max_to_keep=1)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint)
            ckpt_end = 5 + manager.latest_checkpoint.find("ckpt-")
            epoch_start = int(manager.latest_checkpoint[ckpt_end:])
            if self.strategy is None:
                opt_iters = self.optimizer.iterations
            else:
                opt_iters = self.optimizer.iterations.values[0]
            print(
                "Checkpoint restored at epoch %d, optimizer iteration %d."
                % (epoch_start, opt_iters)
            )
        else:
            print("No checkpoints found.")
            epoch_start = 0

        # improvement variables
        valid_best = np.inf
        unimproved = 0

        # training loop
        for ei in range(epoch_start, self.train_epochs_max):
            if ei >= self.train_epochs_min and unimproved > self.patience:
                break
            else:
                # train
                t0 = time.time()
                train_iter = iter(self.train_data[0].dataset)
                # print("self.train_epoch_batches[0]: ", self.train_epoch_batches[0])
                # print("self.train_epoch_batches[0]: ", len(self.train_epoch_batches))
                # print("self.eval_epoch_batches[0]: ", self.eval_epoch_batches[0])
                # print(""+1)
                for si in range(self.train_epoch_batches[0]):
                    
                    if self.steps_per_epoch_max is not None and si >= self.steps_per_epoch_max:
                        break
                    
                    # x, label, exon_mask = None, None, None
                    # if self.train_data[0].has_mask :
                    #     x, label, exon_mask = safe_next(train_iter)
                    # else :
                    #     x, label = safe_next(train_iter)
                    x, label, exon_mask, repeat_mask = None, None, None, None
                    if self.train_data[0].has_mask:
                        x, label, exon_mask, repeat_mask = safe_next(train_iter)
                    else:
                        x, label = safe_next(train_iter)

                    
                    mask_size = tf.cast(self.mask_rate * x.shape[1], dtype=tf.int32)
                    
                    if self.strategy is None:
                        # x_masked, x, ind, sw = prep_mlm(x, label, mask_size, exon_mask=exon_mask, training=True)
                        x_masked, x, ind, sw = prep_mlm(x, label, mask_size, exon_mask=exon_mask, repeat_mask=repeat_mask, training=True)
                        train_step(x_masked, x, ind, sample_weight=sw)
                    
                    if ei == epoch_start and si == 0:
                        print("Successful first step!", flush=True)

                # evaluate
                for x_tuple in self.eval_data[0].dataset:
                    
                    # x, label, exon_mask = None, None, None
                    # if self.eval_data[0].has_mask :
                    #     x, label, exon_mask = x_tuple
                    # else :
                    #     x, label = x_tuple
                    x, label, exon_mask, repeat_mask = None, None, None, None
                    if self.eval_data[0].has_mask:
                        x, label, exon_mask, repeat_mask = x_tuple
                    else:
                        x, label = x_tuple

                    
                    mask_size = tf.cast(self.mask_rate * x.shape[1], dtype=tf.int32)
                    
                    if self.strategy is None:
                        #x_masked, x, ind, sw = prep_mlm(x, label, mask_size, exon_mask=exon_mask, training=False)
                        #eval_step(x_masked, x, ind, sample_weight=sw)
                        
                        eval_loss_repeats = []
                        for _ in range(self.repeat_eval) :
                            
                            # x_masked, x, ind, sw = prep_mlm(x, label, mask_size, exon_mask=exon_mask, training=False)
                            x_masked, x, ind, sw = prep_mlm(x, label, mask_size, exon_mask=exon_mask, repeat_mask=repeat_mask, training=False)
                            eval_loss_repeats.append(eval_step(x_masked, x, ind, sample_weight=sw)[..., None])
                        
                        eval_loss = tf.reduce_mean(tf.concat(eval_loss_repeats, axis=-1), axis=-1)
                        
                        valid_loss(eval_loss)

                n_train_actual = self.train_epoch_batches[0]
                if self.steps_per_epoch_max is not None and self.train_epoch_batches[0] > self.steps_per_epoch_max:
                    n_train_actual = self.steps_per_epoch_max
                
                n_eval_actual = self.eval_epoch_batches[0]
                
                # print training accuracy
                train_loss_epoch = train_loss.result().numpy()
                print(
                    "Epoch %d - %ds - train_loss: %.4f - steps: %d"
                    % (
                        ei,
                        (time.time() - t0),
                        train_loss_epoch,
                        n_train_actual,
                    ),
                    end="",
                )

                # print validation accuracy
                valid_loss_epoch = valid_loss.result().numpy()
                print(
                    " - valid_loss: %.4f - steps: %d"
                    % (valid_loss_epoch, n_eval_actual),
                    end="",
                )

                # checkpoint
                manager.save()
                seqnn_model.save("%s/model_check.h5" % self.out_dir)

                # check best
                valid_best_epoch = valid_loss_epoch
                if valid_best_epoch < valid_best:
                    print(" - best!", end="")
                    unimproved = 0
                    valid_best = valid_best_epoch
                    seqnn_model.save("%s/model_best.h5" % self.out_dir)
                else:
                    unimproved += 1
                print("", flush=True)

                # reset metrics
                train_loss.reset_states()
                valid_loss.reset_states()
                
                # re-init train data if reshuffling records
                if self.shuffle_records :
                    self.train_data[0].make_dataset()

    def make_optimizer(self):
        """Make optimizer object from given parameters."""
        cyclical1 = True
        for lrs_param in [
            "initial_learning_rate",
            "maximal_learning_rate",
            "final_learning_rate",
            "train_epochs_cycle1",
        ]:
            cyclical1 = cyclical1 & (lrs_param in self.params)
        if cyclical1:
            step_size = self.params["train_epochs_cycle1"] * sum(
                self.train_epoch_batches
            )
            initial_learning_rate = self.params.get("initial_learning_rate")
            lr_schedule = Cyclical1LearningRate(
                initial_learning_rate=self.params["initial_learning_rate"],
                maximal_learning_rate=self.params["maximal_learning_rate"],
                final_learning_rate=self.params["final_learning_rate"],
                step_size=step_size,
            )
        else:
            # schedule (currently OFF)
            initial_learning_rate = self.params.get("learning_rate", 0.01)
            if False:
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate,
                    decay_steps=self.params.get("decay_steps", 100000),
                    decay_rate=self.params.get("decay_rate", 0.96),
                    staircase=True,
                )
            else:
                lr_schedule = initial_learning_rate

        if "warmup_steps" in self.params:
            lr_schedule = WarmUp(
                initial_learning_rate=initial_learning_rate,
                warmup_steps=self.params["warmup_steps"],
                decay_schedule=lr_schedule,
            )

        global_clipnorm = self.params.get("global_clipnorm", None)
        if "clip_norm" in self.params:
            clip_norm = self.params["clip_norm"]
        elif "clipnorm" in self.params:
            clip_norm = self.params["clipnorm"]
        else:
            clip_norm = None

        # adaptive gradient clipping handled in fit method
        self.agc_clip = self.params.get("agc_clip", None)

        # optimizer
        optimizer_type = self.params.get("optimizer", "sgd").lower()
        if optimizer_type == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=self.params.get("adam_beta1", 0.9),
                beta_2=self.params.get("adam_beta2", 0.999),
                clipnorm=clip_norm,
                global_clipnorm=global_clipnorm,
                amsgrad=False,
            )  # reduces performance in my experience

        elif optimizer_type == "adamw":
            self.optimizer = tf.keras.optimizers.AdamW(
                weight_decay=self.params.get("weight_decay", 0),
                learning_rate=lr_schedule,
                beta_1=self.params.get("adam_beta1", 0.9),
                beta_2=self.params.get("adam_beta2", 0.999),
                clipnorm=clip_norm,
                global_clipnorm=global_clipnorm,
                amsgrad=False,
            )  # reduces performance in my experience

        elif optimizer_type in ["sgd", "momentum"]:
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=self.params.get("momentum", 0.99),
                clipnorm=clip_norm,
                global_clipnorm=global_clipnorm,
            )

        else:
            print("Cannot recognize optimization algorithm %s" % optimizer_type)
            exit(1)


################################################################
# AGC
# https://github.com/sayakpaul/Adaptive-Gradient-Clipping


def compute_norm(x, axis, keepdims):
    """Compute L2 norm of a tensor across an axis."""
    return tf.math.reduce_sum(x**2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    """Compute L2 norm of a tensor across its last dimension."""
    if len(x.get_shape()) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.get_shape()) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.get_shape()) == 4:  # Conv kernels of shape HWIO
        axis = [
            0,
            1,
            2,
        ]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4]! {x}")
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(
    parameters, gradients, clip_factor: float = 0.1, eps: float = 1e-3
):
    """Adaptive gradient clipping."""
    new_grads = []
    for params, grads in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads


class EarlyStoppingMin(tf.keras.callbacks.EarlyStopping):
    """Stop training when a monitored quantity has stopped improving.

    Args:
      min_epoch: Minimum number of epochs before considering stopping.
    """

    def __init__(self, min_epoch: int = 0, **kwargs):
        super(EarlyStoppingMin, self).__init__(**kwargs)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if epoch >= self.min_epoch and self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)


class Cyclical1LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses cyclical schedule.
    https://yashuseth.blog/2018/11/26/hyper-parameter-tuning-best-practices-learning-rate-batch-size-momentum-weight-decay/

    Args:
      initial_learning_rate (float): The initial learning rate.
      maximal_learning_rate (float): The maximal learning rate after warm up.
      final_learning_rate (float): The final learning rate after cycle.
      step_size (int): Cycle step size.
      name (str, optional): The name of the schedule. Defaults to "Cyclical1LearningRate".
    """

    def __init__(
        self,
        initial_learning_rate: float,
        maximal_learning_rate: float,
        final_learning_rate: float,
        step_size,
        name: str = "Cyclical1LearningRate",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.final_learning_rate = final_learning_rate
        self.step_size = step_size
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "Cyclical1LearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            final_learning_rate = tf.cast(self.final_learning_rate, dtype)

            step_size = tf.cast(self.step_size, dtype)
            cycle = tf.floor(1 + step / (2 * step_size))
            x = tf.abs(step / step_size - 2 * cycle + 1)

            lr = tf.where(
                step > 2 * step_size,
                final_learning_rate,
                initial_learning_rate
                + (maximal_learning_rate - initial_learning_rate)
                * tf.maximum(tf.cast(0, dtype), (1 - x)),
            )
            return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "step_size": self.step_size,
        }


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.
    (h/t HuggingFace.)

    Args:
      initial_learning_rate (:obj:`float`): Initial learning rate after the warmup
        (so this will be the learning rate at the end of the warmup).
      decay_schedule (:obj:`Callable`): The learning rate or schedule function to
        apply after the warmup for the rest of training.
      warmup_steps (:obj:`int`): The number of steps for the warmup part of training.
      power (:obj:`float`, `optional`): Power to use for the polynomial warmup
        (defaults is a linear warmup).
      name (:obj:`str`, `optional`): Optional name prefix for the returned tensors
        during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_schedule: None,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule = decay_schedule
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            if callable(self.decay_schedule):
                warmed_learning_rate = self.decay_schedule(step - self.warmup_steps)
            else:
                warmed_learning_rate = self.decay_schedule
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: warmed_learning_rate,
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule": self.decay_schedule,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def safe_next(data_iter, retry=5, sleep=10):
    attempts = 0
    d = None
    while d is None and attempts < retry:
        try:
            d = next(data_iter)
        except tf.errors.AbortedError:
            print(
                "AbortedError, which has previously indicated NFS daemon restart.",
                file=sys.stderr,
            )
            time.sleep(sleep)
        attempts += 1

    if d is None:
        # let it crash
        d = next(data_iter)
    # for di in range(len(d)):
    #     print("di: ", d[di].shape)
    return d
