from metrics.metrics_depth import Result


def print_metrics(results: Result) -> None:
    print(
        "\n*\n"
        "RMSE={average.rmse:.3f}\n"
        "MAE={average.mae:.3f}\n"
        "Delta1={average.delta1:.3f}\n"
        "Delta2={average.delta2:.3f}\n"
        "Delta3={average.delta3:.3f}\n"
        "REL={average.absrel:.3f}\n"
        "Lg10={average.lg10:.3f}\n"
        "t_GPU={time:.3f}".format(average=results, time=results.gpu_time)
    )
