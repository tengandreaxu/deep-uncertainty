import os

OUTPUT_FOLDER = "saved_models/loss_landscapes"
WS_TRAJECTORY = os.path.join(OUTPUT_FOLDER, "ws_trajectory.dat")

BS_TRAJECTORY = os.path.join(OUTPUT_FOLDER, "bs_trajectory.dat")
WS_MANY = os.path.join(OUTPUT_FOLDER, "ws_many.dat")
BS_MANY = os.path.join(OUTPUT_FOLDER, "bs_many.dat")
WS_BY_EPOCHS = os.path.join(OUTPUT_FOLDER, "ws_by_epochs_many.dat")
BS_BY_EPOCHS = os.path.join(OUTPUT_FOLDER, "bs_by_epochs_many.dat")

VALIDATION_SET = os.path.join(OUTPUT_FOLDER, "valset.pth")
TEST_SET = os.path.join(OUTPUT_FOLDER, "testset.pth")
VALIDATION_OUTPUTS = os.path.join(OUTPUT_FOLDER, "validation_outputs")

SUBSPACE_SAMPLING = os.path.join(OUTPUT_FOLDER, "subspace_sampling")
os.makedirs(SUBSPACE_SAMPLING, exist_ok=True)

# *****************
# Subspace sampling
# *****************
WS_DIAGONAL = os.path.join(SUBSPACE_SAMPLING, "dial_gaussian_whole_Ws.dat")
BS_DIAGONAL = os.path.join(SUBSPACE_SAMPLING, "dial_gaussian_whole_bs.dat")
WS_PCA_GAUSSIAN = os.path.join(SUBSPACE_SAMPLING, "pca_gaussian_whole_Ws.dat")
BS_PCA_GAUSSIAN = os.path.join(SUBSPACE_SAMPLING, "pca_gaussian_whole_bs.dat")
WS_RAND = os.path.join(SUBSPACE_SAMPLING, "rand_Ws.dat")
BS_RAND = os.path.join(SUBSPACE_SAMPLING, "rand_bs.dat")
WS_PCA_RAND_GAUSSIAN = os.path.join(SUBSPACE_SAMPLING, "pca_gaussian_rand_Ws.dat")
BS_PCA_RAND_GAUSSIAN = os.path.join(SUBSPACE_SAMPLING, "pca_gaussian_rand_bs.dat")


ORIG_PRED = os.path.join(SUBSPACE_SAMPLING, "orig_pred.dat")
WA_PRED = os.path.join(SUBSPACE_SAMPLING, "wa_pred.dat")
DIAG_GAUS_PRED = os.path.join(SUBSPACE_SAMPLING, "diag_gaus_pred.dat")
PCA_GAUS_PRED = os.path.join(SUBSPACE_SAMPLING, "pca_gaus_pred.dat")
PCA_RAND_PRED = os.path.join(SUBSPACE_SAMPLING, "pca_rand_pred.dat")
RAND_PRED = os.path.join(SUBSPACE_SAMPLING, "rand_pred.dat")
