import Info from "./components/Info.jsx"
import Inference from "./components/Inference.jsx"

function App() {
  return (
    <div className="grid h-screen w-screen grid-cols-1 md:grid-cols-5 bg-neutral-100 dark:bg-neutral-900 overflow-auto">
      <div className="grid grid-cols-1 grid-rows-1 col-span-1 md:col-span-3 content-center">
        <Info />
      </div>
      <div className="col-span-1 md:col-span-2">
        <Inference />
      </div>
    </div>
  )
}

export default App
