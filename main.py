import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

with open("finaldata.txt", encoding="utf-8") as f:
    data = f.read()

chunks = data.split("\n\n")
chunks = [c for c in chunks if len(c.strip()) > 3]

def parse_chunk(chunk):
    try:
        chunk = chunk.strip()
        sep = chunk.split("\n")

        username = sep[0]
        no_of_posts = int(sep[1].split(" post")[0].replace(",", ""))

        followers_raw = sep[2].split(" follower")[0].replace(",", "")
        if "K" in followers_raw:
            no_of_followers = int(float(followers_raw.replace("K", "")) * 1000)
        elif "M" in followers_raw:
            no_of_followers = int(float(followers_raw.replace("M", "")) * 1000000)
        else:
            no_of_followers = int(followers_raw)

        following_raw = sep[3].split(" following")[0].replace(",", "")
        if "K" in following_raw:
            no_of_following = int(float(following_raw.replace("K", "")) * 1000)
        elif "M" in following_raw:
            no_of_following = int(float(following_raw.replace("M", "")) * 1000000)
        else:
            no_of_following = int(following_raw)

        name = sep[4] if len(sep) > 4 else "Unknown"

        if len(sep) > 5:
            type_of_page = sep[5]
            bio = "\n".join(sep[6:])
        else:
            type_of_page = "Unknown"
            bio = ""

        return {
            "username": username,
            "no_of_posts": no_of_posts,
            "no_of_followers": no_of_followers,
            "no_of_following": no_of_following,
            "name": name,
            "type_of_page": type_of_page,
            "bio": bio
        }
    except:
        return None

all_chunks = []
for chunk in chunks:
    parsed = parse_chunk(chunk)
    if parsed:
        all_chunks.append(parsed)

with open("data.json", "w") as f:
    json.dump(all_chunks, f, indent=4)

print("Data parsed successfully!")

max_posts_user = max(all_chunks, key=lambda x: x['no_of_posts'])
max_followers_user = max(all_chunks, key=lambda x: x['no_of_followers'])
max_following_user = max(all_chunks, key=lambda x: x['no_of_following'])

print("Max posts:", max_posts_user['username'])
print("Max followers:", max_followers_user['username'])
print("Max following:", max_following_user['username'])

df = pd.DataFrame(all_chunks).fillna("Unknown")

top10 = df.sort_values(by="no_of_followers", ascending=False).head(10)

plt.figure()
plt.bar(top10["username"], top10["no_of_followers"])
plt.xticks(rotation=45)
plt.title("Top 10 Users by Followers")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(df["no_of_followers"], df["no_of_following"])
plt.title("Followers vs Following")
plt.xlabel("Followers")
plt.ylabel("Following")
plt.tight_layout()
plt.show()

X = df[["no_of_followers", "no_of_following"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

plt.figure()
plt.scatter(df["no_of_followers"], df["no_of_following"], c=df["cluster"])
plt.title("User Clusters")
plt.xlabel("Followers")
plt.ylabel("Following")
plt.tight_layout()
plt.show()