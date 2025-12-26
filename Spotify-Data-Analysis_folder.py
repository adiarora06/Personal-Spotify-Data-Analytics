#import necessary libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import glob

#load in all streaming history files(JSON)
all_files = glob.glob('Streaming_History_Audio_*.json')
print(f"Found {len(all_files)} streaming history files")

#Combine the loaded files into one data set
all_data = []
for file in all_files:
    print(f"Loading {file}...")
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_data.extend(data)

#convert the combined data into a DataFrame
df = pd.DataFrame(all_data)

print(f"\nLoaded {len(df)} total streams!")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumns available:")
print(df.columns.tolist())

#conversions/title - per my preference(can be changed accordingly)
df['ts'] = pd.to_datetime(df['ts'])
df['minutes_played'] = df['ms_played'] / 60000
df['hours_played'] = df['minutes_played'] / 60

#taking out the time features
df['date'] = df['ts'].dt.date
df['hour'] = df['ts'].dt.hour
df['day_of_week'] = df['ts'].dt.day_name()
df['month'] = df['ts'].dt.to_period('M')
df['year'] = df['ts'].dt.year

# Filter out very short plays (less than 30 seconds - likely skips)
df_full_plays = df[df['minutes_played'] >= 0.5]

print(f"\n📊 BASIC STATS:")
print(f"Total streams: {len(df):,}")
print(f"Full plays (>30 sec): {len(df_full_plays):,}")
print(f"Total listening time: {df['hours_played'].sum():,.0f} hours")
print(f"That's {df['hours_played'].sum() / 24:,.0f} days of music!")

# Top 20 Artists
print(f"\n🎵 TOP 20 ARTISTS:")
top_artists = df_full_plays['master_metadata_album_artist_name'].value_counts().head(20)
print(top_artists)

# Top 20 Songs
print(f"\n🎤 TOP 20 SONGS:")
top_songs = df_full_plays['master_metadata_track_name'].value_counts().head(20)
print(top_songs)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Modern color scheme
primary_color = '#1DB954'  # Spotify green
secondary_color = '#191414'  # Spotify black
accent_color = '#FF6B6B'

print("\n📈 Creating personalized visualizations...")

# 1. Listening by hour of day
fig, ax = plt.subplots(figsize=(14, 7))
hourly = df.groupby('hour')['minutes_played'].sum()
bars = ax.bar(hourly.index, hourly.values, color=primary_color, edgecolor='white', linewidth=1.5)
ax.set_title("Adi's Listening Activity by Hour of Day", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
ax.set_ylabel('Minutes Played', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/listening_by_hour.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: listening_by_hour.png")

# 2. Listening by day of week
fig, ax = plt.subplots(figsize=(13, 7))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily = df.groupby('day_of_week')['minutes_played'].sum().reindex(day_order)
colors = [accent_color if day in ['Saturday', 'Sunday'] else primary_color for day in day_order]
bars = ax.bar(day_order, daily.values, color=colors, edgecolor='white', linewidth=2)
ax.set_title("Adi's Weekly Listening Pattern", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', fontsize=13, fontweight='bold')
ax.set_ylabel('Minutes Played', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/listening_by_day.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: listening_by_day.png")

# 3. Monthly listening trends
fig, ax = plt.subplots(figsize=(16, 7))
monthly = df.groupby('month')['hours_played'].sum()
ax.plot(monthly.index.astype(str), monthly.values, 
        marker='o', linewidth=3, markersize=8, 
        color=primary_color, markerfacecolor=accent_color, 
        markeredgewidth=2, markeredgecolor='white')
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.3, color=primary_color)
ax.set_title("Adi's Music Journey Over Time", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=13, fontweight='bold')
ax.set_ylabel('Hours Played', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('visualizations/monthly_trends.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: monthly_trends.png")

# 4. Top 10 artists horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 9))
top_10_artists = df_full_plays['master_metadata_album_artist_name'].value_counts().head(10)
colors_gradient = plt.cm.viridis(range(len(top_10_artists)))
bars = ax.barh(range(len(top_10_artists)), top_10_artists.values, 
               color=colors_gradient, edgecolor='white', linewidth=2)
ax.set_yticks(range(len(top_10_artists)))
ax.set_yticklabels(top_10_artists.index, fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Plays', fontsize=13, fontweight='bold')
ax.set_title("Adi's Top 10 Most Played Artists", fontsize=18, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_10_artists.values)):
    ax.text(value, i, f'  {value:,} plays', 
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/top_artists.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: top_artists.png")

# 5. BONUS: Top 10 Songs
fig, ax = plt.subplots(figsize=(12, 9))
top_10_songs = df_full_plays['master_metadata_track_name'].value_counts().head(10)
colors_gradient = plt.cm.plasma(range(len(top_10_songs)))
bars = ax.barh(range(len(top_10_songs)), top_10_songs.values, 
               color=colors_gradient, edgecolor='white', linewidth=2)
ax.set_yticks(range(len(top_10_songs)))
ax.set_yticklabels(top_10_songs.index, fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Plays', fontsize=13, fontweight='bold')
ax.set_title("Adi's Top 10 Most Played Songs", fontsize=18, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_10_songs.values)):
    ax.text(value, i, f'  {value:,} plays', 
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/top_songs.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: top_songs.png")

print("\n🎉 All personalized visualizations created for Adi!")
print("📁 Check the 'visualizations/' folder to view them")