# Tasco Maps: Complete AWS Deployment Guide

This guide details the step-by-step process to deploy the Tasco Maps API to AWS ECS Fargate using Terraform and CodePipeline. It covers how to retrieve the necessary credentials, set up configuration variables, use the AWS Console, and trigger the automated pipeline.

---

## Prerequisites & Setup

### Step 1: Set up AWS CLI & Credentials (via AWS Web Console)
Before running Terraform, you must grant it access to your AWS account.

1. Log in to the [AWS Management Console](https://aws.amazon.com/console/).
2. In the top search bar, search for **IAM** (Identity and Access Management) and click it.
3. In the left navigation pane, click **Users**, then click **Create user**.
4. Set a name (e.g., `terraform-deployer`), check **Next**.
5. Select **Attach policies directly**.
6. Check the box for **AdministratorAccess** (necessary for provisioning VPCs, ECS clusters, and Pipelines). Click **Next**, then click **Create user**.
7. Click on your newly created user name from the list.
8. Navigate to the **Security credentials** tab.
9. Scroll down to **Access keys** and click **Create access key**.
10. Choose **Command Line Interface (CLI)**, check the confirmation box, and click **Next**.
11. Add an optional tag (e.g., `Terraform CLI`) and click **Create access key**.
12. **CRITICAL:** Copy the **Access key ID** and the **Secret access key**. Store them safely (you won't be able to see the secret key again).

Next, configure your local environment:
```bash
# Run this in your local terminal
aws configure
```
Input your values when prompted:
- **AWS Access Key ID**: `[Your Access Key ID]`
- **AWS Secret Access Key**: `[Your Secret Access Key]`
- **Default region name**: `us-east-1` (or your preferred region)
- **Default output format**: `json`

---

### Step 2: Generate a GitHub Personal Access Token (PAT)
AWS CodePipeline needs access to your GitHub repository to fetch the code when you push changes.

1. Log in to [GitHub](https://github.com).
2. Click your profile picture in the top-right corner and select **Settings**.
3. Scroll down on the left sidebar and click **Developer settings**.
4. Select **Personal access tokens** -> **Tokens (classic)**.
5. Click **Generate new token** -> **Generate new token (classic)**.
6. Give it a note (e.g., `AWS CodePipeline Tasco Maps`).
7. Set the expiration (e.g., 90 days or No expiration).
8. Select the following scopes:
   - [x] **repo** (Full control of private repositories)
   - [x] **admin:repo_hook** (Full control of repository hooks - required so CodePipeline knows when you push)
9. Scroll to the bottom and click **Generate token**.
10. **CRITICAL:** Copy the token immediately. You will not be able to view it again.

---

## Configuration & Deployment

### Step 3: Configure Terraform Variables
Now, prepare your environment variables for Terraform.

1. Navigate to the Terraform folder:
   ```bash
   cd deploy/aws/terraform
   ```
2. Create a file named `terraform.tfvars` inside that directory. (This file is ignored by `.gitignore` to prevent committing secrets).
3. Open the file and fill in your details:
   ```hcl
   github_repo_owner = "your-github-username-or-org"
   github_repo_name  = "Agentic-AI"
   github_token      = "ghp_yourGitHubPersonalAccessTokenHere..."
   aws_region        = "us-east-1"  # Optional: change if you configured another region
   ```

---

### Step 4: Provision the Infrastructure
Execute Terraform to build the AWS resources.

1. Initialize the directory to download the AWS provider:
   ```bash
   terraform init
   ```
2. Run a dry-run check to verify the resources that will be created:
   ```bash
   terraform plan
   ```
3. Apply the configuration (this will take 5-10 minutes to deploy the network, load balancer, and pipeline):
   ```bash
   terraform apply
   ```
   Type `yes` when prompted to confirm.

4. Once finished, note down the output value labeled `alb_dns_name` (e.g., `tasco-maps-alb-123456789.us-east-1.elb.amazonaws.com`). This is the entry point to your API.

---

## Verifying Deployment via Web Console

After `terraform apply` finishes, the network and services are active, but the container registry is empty. The deployment pipeline will run automatically to compile and push your code.

### 1. Monitor the CI/CD Pipeline
1. Search for **CodePipeline** in the AWS Console.
2. Click on `tasco-maps-pipeline`.
3. You will see three stages:
   - **Source:** Pulls your code from GitHub.
   - **Build:** Triggers CodeBuild to log into ECR, run tests, and package the Docker image. (You can click *Details* to view build logs).
   - **Deploy:** Updates the ECS Task definition and rolls out the update.

### 2. Verify ECS Fargate Tasks
1. Search for **Elastic Container Service** (ECS) in the AWS Console.
2. Click **Clusters** -> `tasco-maps-cluster`.
3. Under the **Services** tab, click `tasco-maps-service`.
4. Check the **Deployments and revisions** tab to see if the deployment has rolled out successfully.
5. Under the **Tasks** tab, you should see a task with a status of `RUNNING`.

### 3. Retrieve logs (if troubleshooting)
1. Inside the ECS Service page, go to the **Configuration and tasks** or **Logs** tab.
2. You can view the stdout/stderr logs streamed directly to CloudWatch logs to verify that the FastAPI server launched correctly on port 7860.

### 4. Test the Endpoint
Once the pipeline shows complete, use `curl` or your browser to query the Application Load Balancer DNS name:
```bash
# Check API Health (should return 200 OK)
curl http://[YOUR_ALB_DNS_NAME]/health

# Perform a test search query
curl -G "http://[YOUR_ALB_DNS_NAME]/v1/search" --data-urlencode "q=cafe gần hồ gươm"
```
