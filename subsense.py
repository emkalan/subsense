# TODO
# - add automatic subtitle fetching; maybe with Google search API? look into env variables and use limits
# - use discretion, CC vs standard subtitles

# import libraries, only ones i actually use
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pysrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter
import numpy as np
from datetime import datetime, timedelta
# import re

nltk.download('vader_lexicon')

# initialize sentiment analyzer, thank you VADER !!!
sia = SentimentIntensityAnalyzer()
class SubtitleAnalyzer:
    def __init__(self, srt_file):
        self.srt_file = srt_file
        self.srt_data = None
        self.results = []

    def load_subtitles(self):
        """Load subtitles from the .srt file."""
        self.srt_data = pysrt.open(self.srt_file)

    def preprocess_text(self, text):
        """Remove extra spaces, and newlines."""
        # text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)   NOT NEEDED !!! HALLELUJAH
        return text.strip().replace('\n', ' ')

    def analyze_sentiment(self, text):
        """Analyze sentiment of the text and return intensity using VADER's SentimentIntensityAnalyzer.
        The absolute positive and negative scores are summed to return intensity."""
        scores = sia.polarity_scores(text)
        intensity = abs(scores['pos']) + abs(scores['neg'])
        return intensity

    def analyze_subtitles(self):
        """Analyze the subtitles; get time information, preprocess, analyze sentiment, and append start_time and intensity to results."""
        self.load_subtitles()
        for sub in self.srt_data:
            start_time = sub.start.to_time()
            text = self.preprocess_text(sub.text)
            intensity = self.analyze_sentiment(text)
            self.results.append((start_time, intensity))

    def get_results(self):
        """Return the results."""
        return self.results

class Plotter:
    def __init__(self, results):
        self.results = results

    def time_to_MMSS(self, seconds):
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f'{minutes:02}:{seconds:02}'

    def moving_average(self, data, window_size):
        """Calculate the moving average of a 1D array.

        Note that the window size is proportional to the amount of data points e.g., length of results.
        The 'valid' mode is used for convolution, which means the output array is necessarily smaller than the input array.
        Edge values are not included in the calculation."""
        return np.convolve(data, np.ones(window_size), 'valid') / window_size # i could have done, Savitzky-Golay. but that is more math.

    def plot_results(self, window_size):
        """Plot the emotional intensity and sentiment, both raw and smoothed using moving_average."""
        times = [datetime.combine(datetime.min, r[0]) for r in self.results]
        intensities = [r[1] for r in self.results]

        numeric_times = [(t - times[0]).total_seconds() for t in times]

        smoothed_intensities = self.moving_average(intensities, window_size)
        smoothed_times = numeric_times[window_size - 1:]

        plt.figure(figsize=(45, 5))
        plt.plot(numeric_times, intensities, label='Emotional Intensity', color='red', marker='o', alpha=0.5)
        plt.plot(smoothed_times, smoothed_intensities, label='Smoothed Intensity', color='blue', linewidth=2)
        plt.xlabel('Time (MM:SS)')
        plt.ylabel('Intensity')
        plt.title('Emotional Intensity by Time')
        plt.legend()
        plt.grid()

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: self.time_to_MMSS(x)))

        plt.show()

def main():
    srt_file = input("Enter the path to the .srt file: ")
    subtitle_analyzer = SubtitleAnalyzer(srt_file)
    subtitle_analyzer.analyze_subtitles()

    results = subtitle_analyzer.get_results()

    window_size = max(3, len(results) // 100) # window size will be at minimum 3, but scale with the length of data points
    # window size will be either 3 or 1% of the length of the data, whichever is larger

    plotter = Plotter(results)
    plotter.plot_results(window_size)

if __name__ == "__main__":
    main()