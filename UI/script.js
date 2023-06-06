const form = document.getElementById("search-form");
const documentList = document.getElementById("document-list");
const loadingIndicator = document.getElementById("loading-indicator");
const searchButton = document.getElementById("search-button");

// Function to render the document list
function renderDocumentList() {
  documentList.innerHTML = ""; // Clear the existing list

  const ul = document.createElement("ul");

  Object.entries(documents).forEach(([doc_id, text]) => {
    const li = document.createElement("li");
    li.textContent = `${doc_id}: ${text}`;
    ul.appendChild(li);
  });

  documentList.appendChild(ul);
}

// Event listener for the form submission
form.addEventListener("submit", function(event) {
  event.preventDefault(); // Prevent form submission

  const selectedDataset = document.getElementById("datasets").value;
  const searchQuery = document.getElementById("search-input").value;

  // Disable the search button
  searchButton.disabled = true;

  // Show loading text on the search button
  searchButton.innerHTML = "Loading...";

  // Prepare the request payload
  const payload = {
    query: searchQuery,
    dataset_name: selectedDataset
  };

  // Send the POST request to the API endpoint using Axios
  axios
    .post("http://localhost:5000/query", payload)
    .then(function(response) {
      documents = response.data.relevant_docs || {};

      // Render the updated document list
      renderDocumentList();

      // Enable the search button and restore its original text
      searchButton.disabled = false;
      searchButton.innerHTML = "Search";

      // Hide the loading indicator
      loadingIndicator.style.display = "none";
    })
    .catch(function(error) {
      console.log(error);

      // Enable the search button and restore its original text
      searchButton.disabled = false;
      searchButton.innerHTML = "Search";

      // Hide the loading indicator
      loadingIndicator.style.display = "none";
    });
});
