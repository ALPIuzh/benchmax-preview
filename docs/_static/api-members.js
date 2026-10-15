// Keep class introductions visible while grouping generated member documentation.
function groupApiMembers() {
  document.querySelectorAll("article dl.py.class > dd, article dl.py.exception > dd").forEach((body) => {
    for (const [label, selector] of [
      ["Properties", "dl.py.attribute, dl.py.property"],
      ["Methods", "dl.py.method, dl.py.classmethod, dl.py.staticmethod"],
    ]) {
      const members = Array.from(body.children).filter((child) => child.matches(selector));
      if (!members.length) continue;
      const group = document.createElement("details");
      group.className = "api-members";
      const summary = document.createElement("summary");
      summary.textContent = label;
      group.append(summary);
      members[0].before(group);
      members.forEach((member) => group.append(member));
    }
  });

  function revealMember() {
    let id;
    try {
      id = decodeURIComponent(window.location.hash.slice(1));
    } catch {
      return;
    }
    const target = document.getElementById(id);
    if (!target) return;
    let opened = false;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.matches("details") && !parent.open) {
        parent.open = true;
        opened = true;
      }
    }
    if (opened) target.scrollIntoView();
  }
  revealMember();
  window.addEventListener("hashchange", revealMember);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", groupApiMembers);
} else {
  groupApiMembers();
}
