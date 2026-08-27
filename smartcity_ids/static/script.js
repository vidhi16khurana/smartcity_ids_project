const runButton = document.getElementById("runDetectionBtn");

if (runButton) {
    runButton.addEventListener("click", runDetection);
}


async function runDetection() {

    const button = document.getElementById("runDetectionBtn");
    const resultsContainer = document.getElementById("resultsContainer");

    button.disabled = true;
    button.innerHTML = "⏳ AI Detection Running...";

    resultsContainer.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🧠</div>

            <h3>Analyzing Chandigarh Smart City</h3>

            <p>
                Network, IoT and Application AI agents are analyzing
                simulated Chandigarh Smart City infrastructure.
            </p>
        </div>
    `;

    try {

        const response = await fetch("/run-detection", {
            method: "POST"
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(
                data.error || "Detection failed"
            );
        }


        // UPDATE STATISTICS

        document.getElementById("totalAlerts").textContent =
            data.total_alerts;

        document.getElementById("criticalThreats").textContent =
            data.critical_threats;

        document.getElementById("totalCampaigns").textContent =
            data.total_campaigns;


        // UPDATE STATUS

        const detectionStatus =
            document.getElementById("detectionStatus");

        detectionStatus.textContent =
            `✓ Analysis completed for ${data.city}`;


        // DISPLAY RESULTS

        displayCampaigns(
            data.campaigns,
            resultsContainer
        );

    } catch (error) {

        console.error(error);

        resultsContainer.innerHTML = `
            <div class="empty-state">

                <div class="empty-icon">
                    ⚠️
                </div>

                <h3>
                    Detection Failed
                </h3>

                <p>
                    ${error.message}
                </p>

            </div>
        `;

    } finally {

        button.disabled = false;

        button.innerHTML = `
            <span>▶</span>
            Run AI Detection
        `;
    }
}


function displayCampaigns(campaigns, container) {

    if (!campaigns || campaigns.length === 0) {

        container.innerHTML = `
            <div class="empty-state">

                <div class="empty-icon">
                    🛡️
                </div>

                <h3>
                    No Coordinated Attack Campaign Detected
                </h3>

                <p>
                    The AI agents did not identify a coordinated
                    multi-layer cyberattack campaign.
                </p>

            </div>
        `;

        return;
    }


    container.innerHTML = campaigns.map(campaign => {

        const layers =
            Array.isArray(campaign.layers_involved)
                ? campaign.layers_involved.join(", ")
                : campaign.layers_involved;

        return `

            <div class="campaign-card">


                <div class="campaign-header">

                    <div>

                        <h3>
                            🚨 ${campaign.campaign_id}
                        </h3>

                        <p>
                            📍 ${campaign.city} |
                            ${campaign.location}
                        </p>

                    </div>


                    <span class="severity-badge ${campaign.severity.toLowerCase()}">

                        ${campaign.severity}

                    </span>

                </div>


                <!-- ATTACK TYPE -->

                <div class="attack-type-box">

                    <span class="attack-label">
                        🚨 DETECTED ATTACK
                    </span>

                    <h4>
                        ${campaign.attack_type}
                    </h4>

                </div>


                <!-- ATTACK DETAILS -->

                <div class="campaign-details">

                    <div>

                        <span>
                            📍 LOCATION
                        </span>

                        <strong>
                            ${campaign.location}, Chandigarh
                        </strong>

                    </div>


                    <div>

                        <span>
                            🎯 TARGET
                        </span>

                        <strong>
                            ${campaign.target}
                        </strong>

                    </div>


                    <div>

                        <span>
                            📊 AI CONFIDENCE
                        </span>

                        <strong>
                            ${campaign.confidence}
                        </strong>

                    </div>


                    <div>

                        <span>
                            🔗 AI AGENTS INVOLVED
                        </span>

                        <strong>
                            ${layers}
                        </strong>

                    </div>

                </div>


                <!-- WHY DETECTED -->

                <div class="why-detected-box">

                    <h4>
                        🧠 Why Was This Detected?
                    </h4>

                    <p>
                        ${campaign.why_detected}
                    </p>

                </div>


                <!-- AI ASSESSMENT -->

                <div class="ai-assessment-box">

                    <h4>
                        🤖 AI Security Assessment
                    </h4>

                    <p>
                        ${campaign.ai_assessment}
                    </p>

                </div>


                <!-- TECHNICAL EVIDENCE -->

                <div class="technical-box">

                    <h4>
                        🔬 Technical Evidence
                    </h4>

                    <p>
                        ${campaign.technical_explanation}
                    </p>

                </div>


            </div>

        `;

    }).join("");
}