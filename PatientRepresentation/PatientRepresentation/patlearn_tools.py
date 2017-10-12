import numpy as np


def r2correlation(model, labels): # TODO: subtract out weighted mean of x and y
    weighted_label_list = []
    weighted_rep_list = []
    sum_weight = 0.
    for patient_id in labels:
        if patient_id not in model.patients:
            continue
        label = labels[patient_id]
        rep = model.patient_reps[patient_id]
        weight = model.weights[patient_id]

        weighted_label_list += [[weight * label]]
        weighted_rep_list += [weight * rep]
        sum_weight += weight
    weighted_y = np.array(weighted_label_list)
    y_mean = np.sum(weighted_y, axis=0)/sum_weight
    weighted_z = np.concatenate(weighted_rep_list, axis=0)
    z_mean = np.sum(weighted_z, axis=0)/sum_weight

    label_list = []
    weighted_label_list = []
    rep_list = []
    weighted_rep_list = []
    for patient_id in labels:
        if patient_id not in model.patients:
            continue
        label = np.array([[labels[patient_id]]]) - y_mean
        rep = model.patient_reps[patient_id] - z_mean
        weight = model.weights[patient_id]

        label_list += [label]
        weighted_label_list += [weight * label]
        rep_list += [rep]
        weighted_rep_list += [weight * rep]

    y = np.concatenate(label_list, axis=0)
    weighted_y = np.concatenate(weighted_label_list, axis=0)
    z = np.concatenate(rep_list, axis=0)
    weighted_z = np.concatenate(weighted_rep_list, axis=0)

    u = y.T.dot(weighted_z)/sum_weight
    if u.dot(u.T) < 1e-12:
        return 0
    
    x = z.dot(u.T)
    weighted_x = weighted_z.dot(u.T)

    cov = x.T.dot(weighted_y)[0,0]
    varx = x.T.dot(weighted_x)[0,0]
    vary = y.T.dot(weighted_y)[0,0]

    print 'x mean: {}'.format(np.mean(weighted_x))
    print 'y mean: {}'.format(np.mean(weighted_y))

    return cov*cov/(varx*vary)


def randomCorrelation(model):
    labels = dict()
    for patient_id in model.patients:
        labels[patient_id] = np.random.randint(2)

    return r2correlation(model, labels)