import Info from "./components/Info.jsx"
import Inference from "./components/Inference.jsx"

function App() {
  return (
    <div className="md:flex md:content-center md:items-center h-screen w-screen bg-neutral-100 dark:bg-neutral-900 overflow-auto">
      <div className="m-auto p-3 grid gap-5 md:gap-8 grid-cols-1 md:grid-rows-6 md:grid-cols-5 bg-neutral-100 dark:bg-neutral-900">
        <Info />
        <Inference />
      </div>
    </div>
  )
}

export default App
