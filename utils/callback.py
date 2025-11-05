from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping


def callbacks(rlr_monitor):
    cbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor=rlr_monitor, factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8),
        EarlyStopping(monitor=rlr_monitor, patience=40, restore_best_weights=True)
    ]

    return cbacks
