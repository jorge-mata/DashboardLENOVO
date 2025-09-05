# Microsoft Forms Direct Integration Guide

## 🚀 Quick Setup Options

### Option 1: Power Automate Webhook (Recommended - Easy)

**Step 1: Create Power Automate Flow**
1. Go to [Power Automate](https://flow.microsoft.com)
2. Create new flow: **"When a new response is submitted"**
3. Select your Microsoft Form

**Step 2: Add HTTP Action**
1. Add action: **HTTP - Post**
2. **URI**: `http://your-server:8501/webhook/forms_data`
3. **Method**: POST
4. **Headers**: `Content-Type: application/json`
5. **Body**: (Copy this JSON, replace question IDs with yours)

```json
{
  "response_id": "@{triggerOutputs()?['body/responseId']}",
  "submitted_time": "@{triggerOutputs()?['body/submissionDateTime']}",
  "responder_email": "@{triggerOutputs()?['body/responder']}",
  "cpu_part_number": "@{triggerOutputs()?['body/r_QUESTION_ID_1']}",
  "cpu_brand": "@{triggerOutputs()?['body/r_QUESTION_ID_2']}",
  "quantity": "@{triggerOutputs()?['body/r_QUESTION_ID_3']}",
  "supplier": "@{triggerOutputs()?['body/r_QUESTION_ID_4']}",
  "delivery_date": "@{triggerOutputs()?['body/r_QUESTION_ID_5']}",
  "po_number": "@{triggerOutputs()?['body/r_QUESTION_ID_6']}"
}
```

**Step 3: Find Your Question IDs**
1. Go to your Form → Responses → Open in Excel
2. Column headers show question IDs (like `r_12a34b56c78d9e0f`)
3. Replace `QUESTION_ID_1`, etc. with your actual IDs

### Option 2: SharePoint List Integration (Medium)

Microsoft Forms automatically saves to SharePoint:

1. Go to your Form → Responses → **"Open in Excel"**
2. Note the SharePoint site URL and list name
3. Use SharePoint REST API or Python `office365` library
4. Connect directly to the list where responses are stored

### Option 3: Microsoft Graph API (Advanced)

**Prerequisites:**
- Azure App Registration
- Forms.Read.All permissions
- Client ID, Secret, Tenant ID

**Setup:**
1. Azure Portal → App Registrations → New
2. API Permissions → Microsoft Graph → Forms.Read.All
3. Certificates & Secrets → New client secret
4. Copy Client ID, Secret, Tenant ID

**Usage:**
```python
# Use the forms_integration.py module
from forms_integration import MicrosoftFormsReader

reader = MicrosoftFormsReader(tenant_id, client_id, client_secret)
df = reader.get_form_responses(form_id)
```

## 🎯 Dashboard Integration

Once set up, your dashboard will show:
- **Data Source dropdown** with "Direct Forms Connection" option
- **Real-time data** from your Microsoft Form
- **Cross-checking** between dummy data and real responses
- **Live KPIs** based on actual CPU orders

## 📊 Form Questions for Best Results

Structure your Microsoft Form with these questions:

1. **"What is the CPU part number?"** (Text)
2. **"Select the CPU brand:"** (Choice: Intel, AMD)
3. **"How many CPUs are needed?"** (Number)
4. **"Supplier name:"** (Text)
5. **"Required delivery date:"** (Date)
6. **"Purchase Order number:"** (Text)
7. **"Origin country:"** (Text)

## 🔄 Testing

1. Submit a test response to your Form
2. Check Power Automate flow execution
3. Verify data appears in dashboard
4. Compare with dummy data for validation

## 🛠️ Troubleshooting

**Common Issues:**
- **Flow not triggering**: Check form permissions
- **Data not appearing**: Verify webhook URL and JSON format
- **Missing columns**: Dashboard will create defaults for missing fields
- **Date format issues**: Use ISO format (YYYY-MM-DD)

**Debug Steps:**
1. Test flow manually in Power Automate
2. Check flow run history for errors
3. Verify JSON structure matches expected format
4. Test with sample data first
