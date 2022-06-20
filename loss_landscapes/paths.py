import os

OUTPUT_FOLDER = "saved_models/loss_landscapes"
WS_TRAJECTORY = os.path.join(OUTPUT_FOLDER, "ws_trajectory.dat")

BS_TRAJECTORY = os.path.join(OUTPUT_FOLDER, "bs_trajectory.dat")
WS_MANY = os.path.join(OUTPUT_FOLDER, "ws_many.dat")
BS_MANY = os.path.join(OUTPUT_FOLDER, "bs_many.dat")

VALIDATION_SET = os.path.join(OUTPUT_FOLDER, "valset.pth")

VALIDATION_OUTPUTS = os.path.join(OUTPUT_FOLDER, "validation_outputs")
