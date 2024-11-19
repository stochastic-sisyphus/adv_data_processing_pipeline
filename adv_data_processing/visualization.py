import matplotlib.pyplot as plt
import seaborn as sns

# Make dependencies optional
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

PLOTTING_AVAILABLE = True

def _check_plotting_available():
    if not PLOTTING_AVAILABLE:
        raise ImportError(
            "Plotting functions require matplotlib and seaborn. "
            "Please install them with: pip install matplotlib seaborn"
        )

def _check_wordcloud_available():
    if not WORDCLOUD_AVAILABLE:
        raise ImportError(
            "Wordcloud functions require wordcloud package. "
            "Please install it with: pip install wordcloud"
        )

def plot_feature_importance(feature_importance):
    """Plot feature importance as a bar chart."""
    _check_plotting_available()
    
    # Sort features by importance
    sorted_features = dict(
        sorted(feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True)
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=list(sorted_features.values()),
        y=list(sorted_features.keys()),
        ax=ax
    )
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    
    return fig

def plot_correlation_matrix(df):
    """Plot correlation matrix heatmap."""
    _check_plotting_available()
    
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        ax=ax
    )
    ax.set_title('Correlation Matrix')
    
    return fig

def plot_distribution(series, target=None):
    """Plot distribution of a series, optionally split by target."""
    _check_plotting_available()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if target is not None:
        for label in target.unique():
            mask = target == label
            sns.kdeplot(
                series[mask],
                label=f'Target={label}',
                ax=ax
            )
        ax.legend()
    else:
        sns.histplot(series, ax=ax)
        
    ax.set_title(f'Distribution of {series.name}')
    
    return fig

def plot_wordcloud(text_data):
    """Plot wordcloud from text data."""
    _check_wordcloud_available()
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white'
    ).generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

