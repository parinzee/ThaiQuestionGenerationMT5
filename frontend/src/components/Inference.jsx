import { useForm } from "react-hook-form";
import Settings from "./Settings";
import ModelInput from "./ModelInput";

function Inference() {
    const { register, handleSubmit } = useForm();
    const onSubmit = data => console.log(data);
    return (
        <form onSubmit={handleSubmit(onSubmit)} className="m-2 md:m-8">
            <ModelInput register={register} />
            <Settings register={register} />
        </form>
    )
}

export default Inference