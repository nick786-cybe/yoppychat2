document.addEventListener('DOMContentLoaded', function() {
    const createPlanForm = document.getElementById('createPlanForm');
    if (createPlanForm) {
        createPlanForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            
            fetch('/api/admin/create_plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                if (result.status === 'success') {
                    showNotification(result.message, 'success');
                    setTimeout(() => window.location.reload(), 1500);
                } else {
                    throw new Error(result.message);
                }
            })
            .catch(error => showNotification(error.toString(), 'error'));
        });
    }
});

function setCurrentPlan(id, type, planId) {
    if (!confirm(`Are you sure you want to set the plan to "${planId}" for this ${type}?`)) {
        return;
    }

    fetch('/api/admin/set_current_plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: id, type: type, plan_id: planId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showNotification(data.message, 'success');
            setTimeout(() => window.location.reload(), 1500);
        } else {
            throw new Error(data.message);
        }
    })
    .catch(error => showNotification(error.toString(), 'error'));
}

function setDefaultPlan(planId, type) {
    if (!confirm(`Are you sure you want to set "${planId}" as the default for new ${type}s?`)) {
        return;
    }

    fetch('/api/admin/set_default_plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan_id: planId, type: type })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showNotification(data.message, 'success');
        } else {
            throw new Error(data.message);
        }
    })
    .catch(error => showNotification(error.toString(), 'error'));
}
function removePlan(id, type) {
    if (!confirm(`Are you sure you want to remove the current plan for this ${type}? This will revert it to the default plan.`)) {
        return;
    }

    fetch('/api/admin/remove_plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: id, type: type })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showNotification(data.message, 'success');
            setTimeout(() => window.location.reload(), 1500);
        } else {
            throw new Error(data.message);
        }
    })
    .catch(error => showNotification(error.toString(), 'error'));
}