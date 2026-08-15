Deploying the NVIDIA Video Search and Summarization (VSS) Blueprint locally on the DGX Spark is the perfect move for a 24-hour sprint. Because the VSS Blueprint uses Docker Compose and NVIDIA NIMs, you can skip the complex environment configurations and immediately get to building your agentic workflow.

Here is the step-by-step guide to get the VSS base vision agent running on your Spark:

## 1. Verify DGX Spark Prerequisites

Before cloning anything, ensure your DGX Spark environment matches these supported versions. Mismatches here—especially with Docker—are the #1 cause of lost time in hackathons:

* **OS:** DGX OS 7.4.0
* **NVIDIA Driver:** 580.95.05
* **Docker Engine:** `28.3.3 <= Docker Engine < 29.5.0` *(Do not ignore this—unsupported versions will fail when pulling NGC images)*.
* **NVIDIA Container Toolkit:** 1.17.8+
* **NGC CLI:** 4.10.0+

## 2. Authentication & Download

You need an NVIDIA GPU Cloud (NGC) API key to pull the heavy containers (like the Cosmos Reasoner Vision-Language Model and the Nemotron LLM).

1. **Configure NGC CLI:**
Generate an API key from your NVIDIA account, then configure your local CLI:

```bash
ngc config set

```

Follow the prompts to enter your API key and set your default organization.


2. **Clone the VSS Repository:**
Pull down the official AI Blueprint repository:

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
cd video-search-and-summarization

```


## 3. Deploy the Developer Profile

The Blueprint includes a "Developer Profile" that is purpose-built for fast experimentation. It automatically spins up the Agent UI, the Video IO & Storage (VIOS) service, and local reasoning NIMs.

1. **Set the Hardware Profile:** Crucial for leveraging the Spark's 128GB memory.
The VSS Blueprint supports the DGX Spark out of the box. In your configuration file (usually `blueprint_config.yml`), ensure your `HARDWARE_PROFILE` is set to the DGX-SPARK profile so the system knows how to allocate resources across the Grace Blackwell superchip.


2. **Launch the Stack:**
Use the provided developer shell script to pull the images and start the containers.

```bash
./scripts/dev-profile.sh up

```

*Note: The initial pull will take a few minutes as it downloads the multi-gigabyte foundation models locally.*


## 4. Test the Vision Agent

Once the deployment script finishes, everything is running completely locally and offline.

Navigate to `http://<HOST_IP>:3000/` in your browser. You will see a web-based chat UI. To verify it works, drag and drop an `.mp4` video (like a 30-second construction site clip) directly into the chat interface.

Once uploaded, try a visual reasoning prompt like: *"Identify any workers in this clip who are missing their safety helmets."* The agent will process the video locally through the VLM and output its reasoning steps before giving you the final answer.
