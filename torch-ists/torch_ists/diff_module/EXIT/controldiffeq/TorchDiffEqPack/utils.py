def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

def extract_keys(options):
    hyperparams = {}
    if 'h' in options.keys():
        hyperparams.update({'h': options['h']})
    if 't0' in options.keys():
        hyperparams.update({'t0': options['t0']})
    if 't1' in options.keys():
        hyperparams.update({'t1':options['t1']})
    if 'rtol' in options.keys():
        hyperparams.update({'rtol': options['rtol']})
    if 'atol' in options.keys():
        hyperparams.update({'atol': options['atol']})
    if 'neval_max' in options.keys():
        hyperparams.update({'neval_max': options['neval_max']})
    if 'print_neval' in options.keys():
        hyperparams.update({'print_neval': options['print_neval']})
    if 'print_direction' in options.keys():
        hyperparams.update({'print_direction': options['print_direction']})
    if 'step_dif_ratio' in options.keys():
        hyperparams.update({'step_dif_ratio': options['step_dif_ratio']})
    if 'safety' in options.keys():
        hyperparams.update({'safety': options['safety']})
    if 'regenerate_graph' in options.keys():
        hyperparams.update({'regenerate_graph': options['regenerate_graph']})
    if 'print_message' in options.keys():
        hyperparams.update({'print_message': options['print_message']})
    if 'print_time' in options.keys():
        hyperparams.update({'print_time': options['print_time']})
    if 'interpolation_method' in options.keys():
        hyperparams.update({'interpolation_method': options['interpolation_method']})
    if 'dense_output' in options.keys():
        hyperparams.update({'dense_output': options['dense_output']})
        print('Dense output mode enabled. The output put is in dense-state and can be called again as usual functions.')

    return hyperparams