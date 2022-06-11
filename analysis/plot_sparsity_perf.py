def transform_by_kernel(results):
  kernel_names = [k['name'] for k in results['results'][0]['runs']]

  # Get sparsities
  sparsities = [r['prop']['sparsity'] for r in results['results']]

  # Get performance per kernel
  def perf(name):
    return [
        run['runtime'] / 1000  # Convert to seconds
        for res in results['results']
        for run in res['runs']
        if run['name'] == name
    ]

  ker_times = [(name, perf(name)) for name in kernel_names]

  return sparsities, ker_times


if __name__ == '__main__':
  import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml

# Load results
with open(sys.argv[1], 'r') as ifs:
  results = yaml.safe_load(ifs.read())

  eq_avg_deg = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

  # Transform data by kernel
  sparsities, ker_times = transform_by_kernel(results)

  # Transform execution time to performance
  num_nodes = results['prop']['num_nodes']

  def num_edges(sparsity):
    return num_nodes * num_nodes * sparsity

  ker_perf = [(name,
               [num_edges(sparse) / t
                for t, sparse in zip(times, sparsities)])
              for name, times in ker_times]

  # Ignore results (very scientific)
  sparsities = sparsities[:-1]
  ker_perf = [(name, perf[:-1]) for name, perf in ker_perf]

  # Plot
  def get_color(name):
    # Get color
    if 'Low' in name:
      color = 'b'
    elif 'Medium' in name:
      color = 'g'
    elif 'High' in name:
      color = 'r'
    else:
      color = 'k'
    return color

  def get_style(name):
    # Get line style
    if 'SM' in name:
      line_style = ':'
      marker = '*'
    elif 'Naive' in name:
      line_style = '-'
      marker = 'D'
    else:
      line_style = '--'
      marker = '^'
    return f'{line_style}{marker}'

  def get_line(name):
    return get_color(name) + get_style(name)

  linewidth = 4
  markersize = 15
  fig, ax = plt.subplots(1, figsize=(15, 14))
  for name, perf in ker_perf:
    ax.plot(sparsities,
            perf,
            get_line(name),
            label=name,
            linewidth=linewidth,
            markersize=markersize)

  ax.set_xticks(sparsities)
  ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

  ax.set_xlabel('Sparsity (larger=more edges/less sparse)', size=25)
  ax.set_xscale('log')

  ax.set_ylabel('Performance (edges/s)', size=25)

  ax.tick_params(axis='both', which='major', labelsize=20)
  ax.tick_params(axis='both', which='minor', labelsize=20)

  category_lines = [
      ax.plot([], [],
              'k' + get_style(name),
              linewidth=linewidth,
              markersize=markersize)[0] for name in ['Naive', '', 'SM']
  ]
  category_legend = ax.legend(category_lines, ["Naive", "Ours", "Ours (SM)"],
                              loc='upper left',
                              fontsize=20,
                              ncol=3,
                              bbox_to_anchor=(0, 0.95))

  color_lines = [
      ax.plot([], [],
              get_color(name) + '-',
              linewidth=linewidth,
              markersize=markersize)[0] for name in ['Low', 'Medium', 'High']
  ]
  color_legend = ax.legend(color_lines,
                           ['Low-Degree', 'Medium-Degree', 'High-Degree'],
                           loc='upper left',
                           fontsize=20,
                           ncol=3)
  ax.add_artist(category_legend)

  plt.title(
      'Aggregation Performance for Varying Sparsities (feature size=1024)',
      size=30,
      x=0.48,
      y=1.02)

  fig.tight_layout()
  #fig.savefig('sparsity.png')
  plt.show()
