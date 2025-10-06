import json
import os

input_path = "data/source"

output_data_path = "data/final_transcripts"

all_contents = []
c = 0

for root, _, files in os.walk(input_path):
    for file in files:
        if file.endswith(".json"):
            json_file_path = os.path.join(root, file)
            with open(json_file_path, "r") as file:
                dataset = json.loads(file.read())
                # print(dataset.keys())
                for key, data in dataset.items():
                    print(key)
                    if not os.path.exists(os.path.join(output_data_path, key)):
                        os.makedirs(os.path.join(output_data_path, key))
                    for session_id, v in data.items():
                        # print(session_id)
                        if "topics" in data[session_id]:
                            topics = data[session_id]["topics"]
                            # print(topics)
                            # print(session_id)
                            c += 1
                            all_contents = {}
                            for topic_title, topic_data in topics.items():
                                # Add a header for each topic to the output for clarity
                                # all_contents.append(f"--- TOPIC {topic_title} ---\n")
                                all_contents[topic_title] = []
                                # print(all_contents)

                                # Check for the 'transcripts' list within each topic [1]
                                if "transcripts" in topic_data:
                                    transcripts = topic_data["transcripts"]

                                    # Iterate through each line item in the transcripts list [2-4]
                                    for line in transcripts:
                                        # **MODIFIED SECTION**
                                        # Check for both 'speaker' and 'contents' keys before processing
                                        if "speaker" in line and "contents" in line:
                                            speaker = line["speaker"]  # [1-4]
                                            contents = line["contents"]  # [1-4]

                                            # Format the string to include the speaker
                                            formatted_line = (
                                                f"'{speaker}': '{contents}'"
                                            )
                                            all_contents[topic_title].append(
                                                formatted_line
                                            )

                            final_json_file_path = os.path.join(
                                output_data_path, key, f"{session_id}.json"
                            )
                            with open(final_json_file_path, "w") as f:
                                json.dump(all_contents, f, indent=4)
                                print(f"{final_json_file_path} dumped")
