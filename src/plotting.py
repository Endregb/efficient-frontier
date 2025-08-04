import matplotlib.pyplot as plt
import seaborn as sns

def set_finance_plot_style():
    """
    Configure matplotlib settings for professional finance plots.
    """
    # Set the font to be more readable and professional
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # Line styles and colors for financial plots
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442',
        '#56B4E9', '#E69F00', '#882255'
    ])
    
    # Grid settings for better readability
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#CCCCCC'
    
    # Figure size and DPI
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    
    # Text sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Seaborn style for better overall aesthetics
    try:
        sns.set_style("whitegrid")
    except ImportError:
        pass
    
    return "Finance plotting style set"