const runButton = document.getElementById("runDetectionBtn");

if (runButton) {
    runButton.addEventListener("click", runDetection);
}


async function runDetection() {

    const button = document.getElementById("runDetectionBtn");
    const resultsContainer = document.getElementById("resultsContainer");
    const detectionStatus = document.getElementById("detectionStatus");

    button.disabled = true;
    button.innerHTML = "⏳ AI Detection Running...";

    detectionStatus.textContent = "Analyzing Chandigarh Smart City...";

    resultsContainer.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🧠</div>

            <h3>Analyzing Chandigarh Smart City</h3>

            <p>
                Network, IoT and Application AI agents are analyzing
                simulated Chandigarh Smart City infrastructure.
                This may take a few seconds.
            </p>
        </div>
    `;

    try {

        const response = await fetch("/run-detection", {
            method: "POST",
            headers: {
                "Accept": "application/json"
            }
        });


        const contentType =
            response.headers.get("content-type") || "";


        if (!contentType.includes("application/json")) {

            const responseText = await response.text();

            console.error(
                "Server returned non-JSON response:",
                responseText
            );

            throw new Error(
                `Server returned an invalid response (HTTP ${response.status}). ` +
                `Please check the Render logs.`
            );
        }


        const data = await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                `Detection failed with HTTP status ${response.status}`
            );
        }


        // UPDATE STATISTICS

        document.getElementById("totalAlerts").textContent =
            data.total_alerts ?? 0;

        document.getElementById("criticalThreats").textContent =
            data.critical_threats ?? 0;

        // IMPORTANT: HTML has id="campaigns"

        document.getElementById("campaigns").textContent =
            data.total_campaigns ?? 0;


        // UPDATE STATUS

        detectionStatus.textContent =
            `✓ Analysis completed for ${data.city}`;


        // DISPLAY RESULTS

        displayCampaigns(
            data.campaigns || [],
            resultsContainer
        );

    } catch (error) {

        console.error("Detection error:", error);

        detectionStatus.textContent =
            "Analysis failed";

        resultsContainer.innerHTML = `
            <div class="empty-state">

                <div class="empty-icon">
                    ⚠️
                </div>

                <h3>
                    Detection Failed
                </h3>

                <p>
                    ${escapeHtml(error.message)}
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
                : (campaign.layers_involved || "Multiple agents");


        const severity =
            campaign.severity || "MEDIUM";


        return `

            <div class="campaign-card">


                <div class="campaign-header">

                    <div>

                        <h3>
                            🚨 ${escapeHtml(campaign.campaign_id || "Campaign")}
                        </h3>

                        <p>
                            📍 ${escapeHtml(campaign.city || "Chandigarh Smart City")} |
                            ${escapeHtml(campaign.location || "Unknown Location")}
                        </p>

                    </div>


                    <span class="severity-badge ${severity.toLowerCase()}">

                        ${escapeHtml(severity)}

                    </span>

                </div>


                <!-- ATTACK TYPE -->

                <div class="attack-type-box">

                    <span class="attack-label">
                        🚨 DETECTED ATTACK
                    </span>

                    <h4>
                        ${escapeHtml(campaign.attack_type || "Suspicious Activity")}
                    </h4>

                </div>


                <!-- ATTACK DETAILS -->

                <div class="campaign-details">

                    <div>

                        <span>
                            📍 LOCATION
                        </span>

                        <strong>
                            ${escapeHtml(campaign.location || "Unknown Location")},
                            Chandigarh
                        </strong>

                    </div>


                    <div>

                        <span>
                            🎯 TARGET
                        </span>

                        <strong>
                            ${escapeHtml(campaign.target || "Smart City Infrastructure")}
                        </strong>

                    </div>


                    <div>

                        <span>
                            📊 AI CONFIDENCE
                        </span>

                        <strong>
                            ${escapeHtml(campaign.confidence || "N/A")}
                        </strong>

                    </div>


                    <div>

                        <span>
                            🔗 AI AGENTS INVOLVED
                        </span>

                        <strong>
                            ${escapeHtml(layers)}
                        </strong>

                    </div>

                </div>


                <!-- WHY DETECTED -->

                <div class="why-detected-box">

                    <h4>
                        🧠 Why Was This Detected?
                    </h4>

                    <p>
                        ${escapeHtml(
                            campaign.why_detected ||
                            "The AI system correlated suspicious activity across the monitored infrastructure."
                        )}
                    </p>

                </div>


                <!-- AI ASSESSMENT -->

                <div class="ai-assessment-box">

                    <h4>
                        🤖 AI Security Assessment
                    </h4>

                    <p>
                        ${escapeHtml(
                            campaign.ai_assessment ||
                            "AI analysis is unavailable."
                        )}
                    </p>

                </div>


                <!-- TECHNICAL EVIDENCE -->

                <div class="technical-box">

                    <h4>
                        🔬 Technical Evidence
                    </h4>

                    <p style="white-space: pre-line;">
                        ${escapeHtml(
                            campaign.technical_explanation ||
                            "No additional technical explanation is available."
                        )}
                    </p>

                </div>


            </div>

        `;

    }).join("");
}


function escapeHtml(value) {

    const div = document.createElement("div");

    div.textContent =
        value === null || value === undefined
            ? ""
            : String(value);

    return div.innerHTML;
}