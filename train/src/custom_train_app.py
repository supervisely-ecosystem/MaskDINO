from supervisely.nn.training.train_app import TrainApp
from supervisely.nn.training.loggers import train_logger
from supervisely import logger
from typing import List


class CustomTrainApp(TrainApp):
    def _setup_logger_callbacks(self):
        """
        Set up callbacks for the training logger.
        """
        epoch_pbar = None
        step_pbar = None

        def start_training_callback(total_epochs: int):
            """
            Callback function that is called when the training process starts.
            """
            nonlocal epoch_pbar
            logger.info(f"Training started for {total_epochs} iterations")
            pbar_widget = self.progress_bar_main
            pbar_widget.show()
            epoch_pbar = pbar_widget(message=f"Iterations", total=total_epochs)

        def finish_training_callback():
            """
            Callback function that is called when the training process finishes.
            """
            self.progress_bar_main.hide()
            self.progress_bar_secondary.hide()
            train_logger.close()

        def start_epoch_callback(total_steps: int):
            """
            Callback function that is called when a new epoch starts.
            """
            nonlocal step_pbar
            logger.info(f"Epoch started. Total steps: {total_steps}")
            pbar_widget = self.progress_bar_secondary
            pbar_widget.show()
            step_pbar = pbar_widget(message=f"Steps", total=total_steps)

        def finish_epoch_callback():
            """
            Callback function that is called when an epoch finishes.
            """
            epoch_pbar.update(1)

        def step_callback():
            """
            Callback function that is called when a step iteration is completed.
            """
            step_pbar.update(1)

        train_logger.add_on_train_started_callback(start_training_callback)
        train_logger.add_on_train_finish_callback(finish_training_callback)

        train_logger.add_on_epoch_started_callback(start_epoch_callback)
        train_logger.add_on_epoch_finish_callback(finish_epoch_callback)

        train_logger.add_on_step_finished_callback(step_callback)

    @property
    def classes(self) -> List[str]:
        """
        Returns the selected classes names for training.

        :return: List of selected classes names.
        :rtype: List[str]
        """
        if not self._has_classes_selector:
            return []
        selected_classes = set(self.gui.classes_selector.get_selected_classes())
        # remap classes with project_meta order
        classes = [
            x for x in self.project_meta.obj_classes.keys() if x in selected_classes
        ]
        if "background" not in classes:
            classes.insert(0, "background")
        return classes
